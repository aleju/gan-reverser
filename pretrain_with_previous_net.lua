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
    --noplot                            Whether to not plot
    --window        (default 23)
    --seed          (default 1)
    --aws                               run in AWS mode
    --saveFreq      (default 50)
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --colorSpace    (default "rgb")     rgb|yuv|hsl|y
    --height           (default 32)           height of images
    --width            (default 32)           width of images
    --G_clamp       (default 5)
    --D_clamp       (default 1)
    --G_L1          (default 0)
    --G_L2          (default 0)
    --D_L1          (default 0)
    --D_L2          (default 1e-4)
    --N_epoch       (default 10000)
    --noiseDim      (default 100)
    --noiseMethod   (default "normal")     normal|uniform
    --network       (default "logs/adversarial.net")
    --N_batches     (default 1000)
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

-- axis of images: 3 channels, <scale> height, <scale> width
if OPT.colorSpace == "y" then
    IMG_DIMENSIONS = {1, OPT.height, OPT.width}
else
    IMG_DIMENSIONS = {3, OPT.height, OPT.width}
end

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
assert(OPT.dataset ~= "NONE")
DATASET.setColorSpace(OPT.colorSpace)
DATASET.setFileExtension("jpg")
DATASET.setHeight(IMG_DIMENSIONS[2])
DATASET.setWidth(IMG_DIMENSIONS[3])
DATASET.setDirs({OPT.dataset})
----------------------------------------------------------------------

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
    -- load previous network
    print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
    local tmp = torch.load(OPT.network)
    D_PREV = tmp.D
    G_PREV = tmp.G
    D_PREV:evaluate()
    G_PREV:evaluate()
    EPOCH = tmp.epoch + 1
    local prevNoiseDim = tmp.opt.noiseDim
    local prevNoiseMethod = tmp.opt.noiseMethod
    local prevColorSpace = tmp.opt.colorSpace
    local prevHeight = tmp.opt.height
    local prevWidth = tmp.opt.width

    if OPT.gpu == false then
        D_PREV:float()
        G_PREV:float()
    end

    -- Initialize G in autoencoder form
    -- G is a Sequential that contains (1) G Encoder and (2) G Decoder (both again Sequentials)
    D = MODELS.create_D(IMG_DIMENSIONS, OPT.gpu ~= false)
    G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)

    print("G_PREV:")
    print(G_PREV)
    print("G:")
    print(G)

    print("D_PREV:")
    print(D_PREV)
    print("D:")
    print(D)

    print(string.format('Number of free parameters in G_PREV: %d', NN_UTILS.getNumberOfParameters(G_PREV)))
    print(string.format('Number of free parameters in G: %d', NN_UTILS.getNumberOfParameters(G)))
    print(string.format('Number of free parameters in D_PREV: %d', NN_UTILS.getNumberOfParameters(D_PREV)))
    print(string.format('Number of free parameters in D: %d', NN_UTILS.getNumberOfParameters(D)))

    -- Mean squared error criterion
    CRITERION_G = nn.MSECriterion()
    CRITERION_D = nn.BCECriterion()

    -- Get parameters and gradients
    PARAMETERS_G, GRAD_PARAMETERS_G = G:getParameters()
    PARAMETERS_D, GRAD_PARAMETERS_D = D:getParameters()

    -- Initialize adam state
    OPTSTATE = {adam={G={}, D={}}}

    if NORMALIZE then
        TRAIN_DATA = DATASET.loadRandomImages(10000)
        NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
    end

    -- initialize noise dims
    local nbBatches = OPT.N_batches
    print("Initializing noiseTensor...")
    local prevNoiseTensor = NN_UTILS.createNoiseInputs(OPT.batchSize*nbBatches, prevNoiseDim, prevNoiseMethod) --torch.Tensor(OPT.batchSize * nbBatches, prevNoiseDim)
    local noiseTensor = NN_UTILS.createNoiseInputs(OPT.batchSize*nbBatches, OPT.noiseDim, OPT.noiseMethod) --torch.Tensor(OPT.batchSize * nbBatches, OPT.noiseDim)
    --prevNoiseTensor:uniform(-1, 1)
    --noiseTensor:uniform(-1, 1)
    for i=1,OPT.batchSize*nbBatches do
        for j=1,math.min(OPT.noiseDim, prevNoiseDim) do
            noiseTensor[i][j] = prevNoiseTensor[i][j]
        end
    end

    for i=1,nbBatches do
        local batchStart = 1 + (i-1) * OPT.batchSize
        local batchEnd = math.min(i * OPT.batchSize, noiseTensor:size(1))
        local prevBatchNoise = prevNoiseTensor[{{batchStart, batchEnd}, {}}]
        local batchNoise = noiseTensor[{{batchStart, batchEnd}, {}}]
        local imagesByGprev = G_PREV:forward(prevBatchNoise):clone()
        imagesByGprev = NN_UTILS.switchColorSpace(imagesByGprev, prevColorSpace, OPT.colorSpace)
        local imagesByG = G:forward(batchNoise):clone()

        local imagesDinput = torch.Tensor(OPT.batchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local imgsDataset = DATASET.loadRandomImages(OPT.batchSize / 2)
        local nAdded = 0
        for j=1,imgsDataset:size() do
            imagesDinput[1 + nAdded] = imgsDataset[j]
            nAdded = nAdded + 1
        end
        for j=1,OPT.batchSize/2 do
            imagesDinput[1 + nAdded] = imagesByGprev[j]
            nAdded = nAdded + 1
        end
        --DISP.image(NN_UTILS.toRgb(imagesDinput, OPT.colorSpace), {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="D input images (batch " .. i .. ")"})
        local predsByDprev = D_PREV:forward(NN_UTILS.switchColorSpace(imagesDinput, OPT.colorSpace, prevColorSpace)):clone()
        local predsByD = D:forward(imagesDinput):clone()

        local fevalG = function(x)
            if x ~= PARAMETERS_G then PARAMETERS_G:copy(x) end
            GRAD_PARAMETERS_G:zero()

            -- forward pass
            local f = CRITERION_G:forward(imagesByG, imagesByGprev)

            -- backward pass
            local df_do = CRITERION_G:backward(imagesByG, imagesByGprev)
            G:backward(batchNoise, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G:add(torch.sign(PARAMETERS_G):mul(OPT.G_L1) + PARAMETERS_G:clone():mul(OPT.G_L2) )
            end

            -- Clamp G's gradients
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end

            return f,GRAD_PARAMETERS_G
        end

        local fevalD = function(x)
            if x ~= PARAMETERS_D then PARAMETERS_D:copy(x) end
            GRAD_PARAMETERS_D:zero()

            --  forward pass
            local f = CRITERION_D:forward(predsByD, predsByDprev)

            -- backward pass
            local df_do = CRITERION_D:backward(predsByD, predsByDprev)
            D:backward(imagesDinput, df_do)

            -- penalties (L1 and L2):
            if OPT.D_L1 ~= 0 or OPT.D_L2 ~= 0 then
                -- Loss:
                f = f + OPT.D_L1 * torch.norm(PARAMETERS_D, 1)
                f = f + OPT.D_L2 * torch.norm(PARAMETERS_D, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_D:add(torch.sign(PARAMETERS_D):mul(OPT.D_L1) + PARAMETERS_D:clone():mul(OPT.D_L2) )
            end

            -- Clamp G's gradients
            if OPT.D_clamp ~= 0 then
                GRAD_PARAMETERS_D:clamp((-1)*OPT.D_clamp, OPT.D_clamp)
            end

            return f,GRAD_PARAMETERS_D
        end

        optim.adam(fevalG, PARAMETERS_G, OPTSTATE.adam.G)
        optim.adam(fevalD, PARAMETERS_D, OPTSTATE.adam.D)

        print(string.format("<batch %d of %d (%.2f%%)> loss G: %.4f, loss D: %.4f", i, nbBatches, 100*i/nbBatches, CRITERION_G.output, CRITERION_D.output))
        if i % 10 == 0 then
            visualizeProgress(i)
        end

        if i % OPT.saveFreq == 0 then
            save()
        end

        --xlua.progress(i, nbBatches)
        collectgarbage()
    end

    save()
end

function save()
    print("Saving networks...")
    local filename = paths.concat(OPT.save, string.format('pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
    NN_UTILS.prepareNetworkForSave(G)
    NN_UTILS.prepareNetworkForSave(D)
    torch.save(filename, {G=G, D=D, opt=OPT})
end

-- Function to plot the current autoencoder training progress,
-- i.e. show training images and images after encode-decode
function visualizeProgress(batchIdx)
    -- deactivate dropout
    G:evaluate()
    D:evaluate()

    local noise = NN_UTILS.createNoiseInputs(100) --torch.Tensor(100, OPT.noiseDim)
    --noise:uniform(-1, 1)
    local images = NN_UTILS.forwardBatched(G, noise, OPT.batchSize):clone()

    local imagesReal = DATASET.loadRandomImages(50)
    local bothImages = torch.Tensor(100, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,imagesReal:size() do
        bothImages[i] = imagesReal[i]
    end
    for i=imagesReal:size()+1,bothImages:size(1) do
        --print(i-imagesReal:size(), imagesReal:size(), images:size(), i)
        bothImages[i] = images[i-imagesReal:size()]
    end
    MODEL_D = D
    local imagesGood, scoreGood = NN_UTILS.sortImagesByPrediction(bothImages, false, 50)
    local imagesBad, scoreBad = NN_UTILS.sortImagesByPrediction(bothImages, true, 50)
    --[[
    for i=1,50 do
        print(scoreGood[i])
        print(scoreBad[i])
    end
    --]]

    -- display images, images after encode-decode, plot of loss function
    DISP.image(NN_UTILS.toRgb(images, OPT.colorSpace), {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="Images by G (batch " .. batchIdx .. ")"})
    DISP.image(NN_UTILS.toRgb(imagesGood, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Images rated as good by D (batch " .. batchIdx .. ")"})
    DISP.image(NN_UTILS.toRgb(imagesBad, OPT.colorSpace), {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title="Images rated as bad by D (batch " .. batchIdx .. ")"})

    -- reactivate dropout
    G:training()
    D:training()
end

main()
