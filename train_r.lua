require 'torch'
require 'image'
require 'paths'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'optim'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'

OPT = lapp[[
    --save          (default "logs")
    --batchSize     (default 32)
    --nbBatches     (default -1)        Max number of batches, <0 is infinite
    --noplot                            Whether to not plot
    --window        (default 23)
    --seed          (default 1)
    --saveFreq      (default 2000)
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --R_clamp       (default 1)
    --R_L1          (default 0)
    --R_L2          (default 1e-4)
    --G             (default "logs/adversarial.net")
    --continue      (default "")
    --dataset       (default "NONE")       Directory that contains *.jpg images
]]

NORMALIZE = false
START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- possible output of disciminator
CLASSES = {"0", "1"}
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- run on gpu if chosen
-- We have to load all kinds of libraries here, otherwise we risk crashes when loading
-- saved networks afterwards
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
require 'dpnn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')

function main()
    -- load G
    print(string.format("<trainer> loading trained G from file '%s'", OPT.G))
    local tmp = torch.load(OPT.G)
    MODEL_G = tmp.G
    MODEL_G:evaluate()
    OPT.noiseDim = tmp.opt.noiseDim
    OPT.noiseMethod = tmp.opt.noiseMethod
    OPT.height = tmp.opt.height
    OPT.width = tmp.opt.width
    OPT.colorSpace = tmp.opt.colorSpace

    if OPT.gpu == false then
        MODEL_G:float()
    end

    ----------------------------------------------------------------------
    -- set stuff dependent on height, width and colorSpace
    ----------------------------------------------------------------------
    -- axis of images: 3 channels, <scale> height, <scale> width
    if OPT.colorSpace == "y" then
        IMG_DIMENSIONS = {1, OPT.height, OPT.width}
    else
        IMG_DIMENSIONS = {3, OPT.height, OPT.width}
    end

    -- get/create dataset
    assert(OPT.dataset ~= "NONE")
    DATASET.setColorSpace(OPT.colorSpace)
    DATASET.setFileExtension("jpg")
    DATASET.setHeight(IMG_DIMENSIONS[2])
    DATASET.setWidth(IMG_DIMENSIONS[3])
    DATASET.setDirs({OPT.dataset})
    ----------------------------------------------------------------------

    -- Initialize G in autoencoder form
    -- G is a Sequential that contains (1) G Encoder and (2) G Decoder (both again Sequentials)
    if OPT.continue ~= "" then
        local tmp = torch.load(OPT.continue)
        MODEL_R = tmp.R
        if OPT.gpu == false then MODEL_R:float() end
    else
        MODEL_R = MODELS.create_R(IMG_DIMENSIONS, OPT.noiseDim, OPT.noiseMethod, OPT.gpu ~= false)
    end

    print("G:")
    print(MODEL_G)

    print("R:")
    print(MODEL_R)

    print(string.format('Number of free parameters in G: %d', NN_UTILS.getNumberOfParameters(MODEL_G)))
    print(string.format('Number of free parameters in R: %d', NN_UTILS.getNumberOfParameters(MODEL_R)))

    -- Mean squared error criterion
    CRITERION_R = nn.MSECriterion()

    -- Get parameters and gradients
    PARAMETERS_R, GRAD_PARAMETERS_R = MODEL_R:getParameters()

    -- Initialize adam state
    OPTSTATE = {adam={R={}}}

    PLOT_DATA = {}
    local losses = {}

    local batchIdx = 1
    while true do
        if OPT.nbBatches >= 0 and batchIdx > OPT.nbBatches then
            break
        end

        local noise = NN_UTILS.createNoiseInputs(OPT.batchSize)
        local images = MODEL_G:forward(noise):clone()

        local fevalR = function(x)
            if x ~= PARAMETERS_R then PARAMETERS_R:copy(x) end
            GRAD_PARAMETERS_R:zero()

            --  forward pass
            local predsByR = MODEL_R:forward(images:clone()):clone()
            local f = CRITERION_R:forward(predsByR, noise)

            -- backward pass
            local df_do = CRITERION_R:backward(predsByR, noise)
            MODEL_R:backward(images, df_do)

            -- penalties (L1 and L2):
            if OPT.R_L1 ~= 0 or OPT.R_L2 ~= 0 then
                -- Loss:
                f = f + OPT.R_L1 * torch.norm(PARAMETERS_R, 1)
                f = f + OPT.R_L2 * torch.norm(PARAMETERS_R, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_R:add(torch.sign(PARAMETERS_R):mul(OPT.R_L1) + PARAMETERS_R:clone():mul(OPT.R_L2))
            end

            -- Clamp G's gradients
            if OPT.R_clamp ~= 0 then
                GRAD_PARAMETERS_R:clamp((-1)*OPT.R_clamp, OPT.R_clamp)
            end

            return f,GRAD_PARAMETERS_R
        end

        optim.adam(fevalR, PARAMETERS_R, OPTSTATE.adam.R)

        if OPT.nbBatches < 0 then
            print(string.format("[batch %d] loss R=%.4f", batchIdx, CRITERION_R.output))
        else
            print(string.format("[batch %d of %d (%.2f%%)] loss R=%.4f", batchIdx, OPT.nbBatches, 100*batchIdx/OPT.nbBatches, CRITERION_R.output))
        end

        if batchIdx % 100 == 0 then
            local attri = MODEL_R:forward(images:clone()):clone()
            print("Example:")
            print(string.format("Noise for G: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f", noise[1][1], noise[1][2], noise[1][3], noise[1][4], noise[1][5], noise[1][6], noise[1][7], noise[1][8], noise[1][9], noise[1][10]))
            print(string.format("Result by R: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f", attri[1][1], attri[1][2], attri[1][3], attri[1][4], attri[1][5], attri[1][6], attri[1][7], attri[1][8], attri[1][9], attri[1][10]))
        end

        if batchIdx % OPT.saveFreq == 0 then
            save()
        end

        MODEL_R:evaluate()

        table.insert(losses, CRITERION_R.output)
        local plotChartEvery = 100
        if batchIdx % plotChartEvery == 0 then
            local low = losses[#losses]
            local avg = 0
            local high = 0
            for i=#losses-plotChartEvery+1,#losses do
                if losses[i] < low then low = losses[i] end
                avg = avg + losses[i]
                if losses[i] > high then high = losses[i] end
            end
            avg = avg / plotChartEvery
            table.insert(PLOT_DATA, {batchIdx, low, avg, high})
            DISP.plot(PLOT_DATA, {win=OPT.window+2, labels={'epoch', 'R loss (low)', 'R loss (avg)', 'R loss (high)'}, title='R Loss'})
        end

        if batchIdx % 25 == 0 then
            local afterR = MODEL_R:forward(images:clone()):clone()
            local afterRafterG = MODEL_G:forward(afterR):clone()
            local imagesGandR = torch.Tensor(afterRafterG:size(1)*2, afterRafterG:size(2), afterRafterG:size(3), afterRafterG:size(4))
            local j = 1
            for i=1,afterRafterG:size(1) do
                imagesGandR[j] = images[i]
                imagesGandR[j+1] = afterRafterG[i]
                j = j + 2
            end
            DISP.image(NN_UTILS.toRgb(imagesGandR, OPT.colorSpace), {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="G and G->R->G (batch " .. batchIdx .. ")"})
        end

        DISP.image(NN_UTILS.toRgb(images, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Generated batch (batch " .. batchIdx .. ")"})

        MODEL_R:training()

        batchIdx = batchIdx + 1
    end
end

function save()
    print("Saving networks...")
    local filename = paths.concat(OPT.save, string.format('r_%dx%dx%d_nd%d_%s.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim, OPT.noiseMethod))
    NN_UTILS.prepareNetworkForSave(MODEL_R)
    torch.save(filename, {R=MODEL_R, opt=OPT})
end

--------
main()
