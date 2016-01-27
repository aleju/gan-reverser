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
    --save             (default "logs")       subdirectory to save logs
    --saveFreq         (default 30)           save every saveFreq epochs
    --epochs           (default -1)           Stop after that epoch
    --network          (default "")           reload pretrained network
    --noplot                                  plot while training
    --batchSize        (default 128)          batch size
    --N_epoch          (default 30)           Number of batches per epoch
    --G_L1             (default 0)            L1 penalty on the weights of G
    --G_L2             (default 0e-6)         L2 penalty on the weights of G
    --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
    --G_optmethod      (default "adam")       adam|adagrad
    --threads          (default 4)            number of threads
    --gpu              (default 0)            gpu to run on (default cpu)
    --noiseDim         (default 100)          dimensionality of noise vector
    --noiseMethod      (default "normal")     normal|uniform
    --window           (default 1)            window id of sample image
    --seed             (default 1)            seed for the RNG
    --colorSpace       (default "rgb")        rgb|yuv|hsl|y
    --height           (default 32)           Height of the training images
    --width            (default 32)           Width of the training images
    --dataset          (default "NONE")       Directory that contains *.jpg images
    --nopretraining
]]

NORMALIZE = false

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

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
    -- Initialize G in autoencoder form
    -- G is a Sequential that contains (1) G Encoder and (2) G Decoder (both again Sequentials)
    G_ENCODER = MODELS.create_G_encoder(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
    G_DECODER = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
    G_AUTOENCODER = nn.Sequential()
    G_AUTOENCODER:add(G_ENCODER)
    G_AUTOENCODER:add(G_DECODER)

    print("G autoencoder:")
    print(G_AUTOENCODER)
    print(string.format('Number of free parameters in G (total): %d', NN_UTILS.getNumberOfParameters(G_AUTOENCODER)))

    -- Mean squared error criterion
    CRITERION = nn.MSECriterion()

    -- Get parameters and gradients
    PARAMETERS_G_AUTOENCODER, GRAD_PARAMETERS_G_AUTOENCODER = G_AUTOENCODER:getParameters()

    -- Initialize adam state
    OPTSTATE = {adam={}}

    if NORMALIZE then
        TRAIN_DATA = DATASET.loadRandomImages(10000)
        NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
    end

    -- training loop
    EPOCH = 1
    while true do
        if OPT.epochs > -1 and OPT.epochs > EPOCH then
            print("<trainer> Last epoch reached.")
            break
        end

        print(string.format("<trainer> Epoch %d", EPOCH))
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch * OPT.batchSize)
        if NORMALIZE then
            TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        end

        epoch()

        if not OPT.noplot then
            visualizeProgress()
        end
    end
end

-- Train G (in autoencoder form) for one epoch
function epoch()
    for batchIdx=1,OPT.N_epoch do
        -- size of this batch, usually OPT.batchSize, may be smaller at the end
        local batchStart = (batchIdx-1) * OPT.batchSize + 1
        local batchEnd = batchStart + OPT.batchSize
        local inputs = torch.Tensor(OPT.batchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(OPT.batchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        --TRAIN_DATA[{{batchStart,batchEnd}, {}, {}, {}}]:clone()
        --local targets = inputs:clone()

        for i=1,OPT.batchSize do
            inputs[i] = TRAIN_DATA[batchStart+i-1]:clone()
            targets[i] = TRAIN_DATA[batchStart+i-1]:clone()
        end

        -- evaluation function for G
        local fevalG = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G_AUTOENCODER then PARAMETERS_G_AUTOENCODER:copy(x) end
            GRAD_PARAMETERS_G_AUTOENCODER:zero()

            --  forward pass
            local outputs = G_AUTOENCODER:forward(inputs)
            local f = CRITERION:forward(outputs, targets)
            showBatch(inputs, OPT.window+3, string.format("Pretraining batch for G (batchIdx=%d, epoch=%d)", batchIdx, EPOCH))

            -- backward pass
            local df_do = CRITERION:backward(outputs, targets)
            G_AUTOENCODER:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G_AUTOENCODER, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G_AUTOENCODER, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G_AUTOENCODER:add(torch.sign(PARAMETERS_G_AUTOENCODER):mul(OPT.G_L1) + PARAMETERS_G_AUTOENCODER:clone():mul(OPT.G_L2) )
            end

            -- Clamp G's gradients
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G_AUTOENCODER:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end

            return f,GRAD_PARAMETERS_G_AUTOENCODER
        end

        -- use Adam as optimizer
        optim.adam(fevalG, PARAMETERS_G_AUTOENCODER, OPTSTATE.adam)

        xlua.progress(batchIdx, OPT.N_epoch)
    end

    print(string.format("<trainer> last batch loss: %.4f", CRITERION.output))

    -- save the model
    if EPOCH % OPT.saveFreq == 0 then
        -- filename is "g_pretrained_CHANNELSxHEIGHTxWIDTH_NOISEDIM.net"
        -- where NOISEDIM is equal to the size of layer between encoder and decoder (z)
        local filename = paths.concat(OPT.save, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        print(string.format("<trainer> saving network to %s", filename))

        -- Clone the autoencoder and deactivate cuda mode
        NN_UTILS.prepareNetworkForSave(G_AUTOENCODER)
        --local G2 = G_AUTOENCODER:clone()
        --G2:float()
        --G2 = NN_UTILS.deactivateCuda(G2)

        -- :get(2) because we only want the decode part
        torch.save(filename, {G=G_AUTOENCODER:get(2), opt=OPT, EPOCH=EPOCH+1})
    end

    EPOCH = EPOCH + 1
end

function showBatch(images, windowId, title, width)
    title = title or string.format("Batch at Epoch %d", EPOCH)
    width = width or IMG_DIMENSIONS[3] * 15
    DISP.image(NN_UTILS.toRgb(images, OPT.colorSpace), {win=windowId, width=width, title=title})
end

-- Function to plot the current autoencoder training progress,
-- i.e. show training images and images after encode-decode
function visualizeProgress()
    -- deactivate dropout
    G_AUTOENCODER:evaluate()

    -- This global static array will be used to save the loss function values
    if PLOT_DATA == nil then PLOT_DATA = {} end

    -- Load some images
    -- we will only test here on potential training images
    local imagesReal = DATASET.loadRandomImages(100)
    if NORMALIZE then
        imagesReal.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    end

    -- Convert them to a tensor (instead of list of tensors),
    -- :forward() and display (DISP) want that
    local imagesRealTensor = torch.Tensor(imagesReal:size(), IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,imagesReal:size() do imagesRealTensor[i] = imagesReal[i] end

    -- encode-decode the images
    local imagesAfterG = G_AUTOENCODER:forward(imagesRealTensor)

    -- log the loss of the last encode-decode
    table.insert(PLOT_DATA, {EPOCH, CRITERION.output})

    -- display images, images after encode-decode, plot of loss function
    DISP.image(NN_UTILS.toRgb(imagesRealTensor, OPT.colorSpace), {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="Original images (before Autoencoder) (EPOCH " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(imagesAfterG, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Images after autoencoder G (EPOCH " .. EPOCH .. ")"})
    DISP.plot(PLOT_DATA, {win=OPT.window+2, labels={'epoch', 'G Loss'}, title='G Loss'})

    -- reactivate dropout
    G_AUTOENCODER:training()
end

main()
