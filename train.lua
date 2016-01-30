require 'torch'
require 'image'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial'
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")       subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --epochs           (default -1)           Stop after that epoch
  --network          (default "")           Filename of a previous training run to continue (in directory --save)
  --G_pretrained_dir (default "logs")       Directory in which pretrained networks may be saved
  --nopretraining                           Whether to deactivate loading of pretrained networks
  --noplot                                  plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  --batchSize        (default 32)           batch size
  --N_epoch          (default 30)           Number of batches per epoch
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 0e-7)         L1 penalty on the weights of D
  --D_L2             (default 1e-4)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       sgd|adagrad|adadelta|adamax|adam|rmsprob
  --G_optmethod      (default "adam")       sgd|adagrad|adadelta|adamax|adam|rmsprob
  --threads          (default 4)            number of threads
  --gpu              (default 0)            gpu to run on (default cpu)
  --noiseDim         (default 32)           dimensionality of noise vector
  --noiseMethod      (default "normal")     normal|uniform
  --window           (default 3)            window id of sample image
  --seed             (default 1)            seed for the RNG
  --colorSpace       (default "rgb")        rgb|yuv|hsl|y
  --height           (default 32)           Height of the training images
  --width            (default 32)           Width of the training images
  --dataset          (default "NONE")       Directory that contains *.jpg images
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
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------

    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        local tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        EPOCH = tmp.epoch + 1
        VIS_NOISE_INPUTS = tmp.vis_noise_inputs
        if NORMALIZE then
            NORMALIZE_MEAN = tmp.normalize_mean
            NORMALIZE_STD = tmp.normalize_std
        end

        if OPT.gpu == false then
            MODEL_D:float()
            MODEL_G:float()
        end
    else
        local pt_filename = paths.concat(OPT.save, string.format('pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        -- pretrained via pretrain_with_previous_net.lua ?
        if not OPT.nopretraining and paths.filep(pt_filename) then
            local tmp = torch.load(pt_filename)
            MODEL_D = tmp.D
            MODEL_G = tmp.G
            MODEL_D:training()
            MODEL_G:training()
            if OPT.gpu == false then
                MODEL_D:float()
                MODEL_G:float()
            end
        else
            --------------
            -- D
            --------------
            MODEL_D = MODELS.create_D(IMG_DIMENSIONS, OPT.gpu ~= false)

            --------------
            -- G
            --------------
            local g_pt_filename = paths.concat(OPT.G_pretrained_dir, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
            if not OPT.nopretraining and paths.filep(g_pt_filename) then
                -- Load a pretrained version of G
                print("<trainer> loading pretrained G...")
                local tmp = torch.load(g_pt_filename)
                MODEL_G = tmp.G
                MODEL_G:training()
                if OPT.gpu == false then
                    MODEL_G:float()
                end
            else
                print("<trainer> Note: Did not find pretrained G")
                MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
            end
        end
    end

    print(MODEL_G)
    print(MODEL_D)

    -- count free parameters in D/G
    print(string.format('Number of free parameters in D: %d', NN_UTILS.getNumberOfParameters(MODEL_D)))
    print(string.format('Number of free parameters in G: %d', NN_UTILS.getNumberOfParameters(MODEL_G)))

    -- loss function: negative log-likelihood
    CRITERION = nn.BCECriterion()

    -- retrieve parameters and gradients
    PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
    PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- Set optimizer states
    OPTSTATE = {
        adagrad = { D = {}, G = {} },
        adadelta = { D = {}, G = {} },
        adamax = { D = {}, G = {} },
        adam = { D = {}, G = {} },
        rmsprop = { D = {}, G = {} },
        sgd = {
            D = {learningRate = OPT.D_sgd_lr, momentum = OPT.D_sgd_momentum},
            G = {learningRate = OPT.G_sgd_lr, momentum = OPT.G_sgd_momentum}
        }
    }

    if NORMALIZE then
        if NORMALIZE_MEAN == nil then
            TRAIN_DATA = DATASET.loadRandomImages(10000)
            NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
        end
    end

    EPOCH = EPOCH or 1
    PLOT_DATA = {}
    VIS_NOISE_INPUTS = VIS_NOISE_INPUTS or NN_UTILS.createNoiseInputs(100)

    -- training loop
    while true do
        if OPT.epochs > -1 and OPT.epochs > EPOCH then
            print("<trainer> Last epoch reached.")
            save()
            break
        end

        local nbLoad = (OPT.N_epoch * OPT.batchSize / 2) * OPT.D_iterations
        print(string.format("<trainer> Loading %d new training images...", nbLoad))
        TRAIN_DATA = DATASET.loadRandomImages(nbLoad)
        if NORMALIZE then
            TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        end

        -- Show images and plots if requested
        if not OPT.noplot then
            visualizeProgress(VIS_NOISE_INPUTS)
        end

        -- Train D and G
        -- ... but train D only while having an accuracy below OPT.D_maxAcc
        --     over the last math.max(20, math.min(1000/OPT.batchSize, 250)) batches
        ADVERSARIAL.train(TRAIN_DATA)

        -- Save current net
        if EPOCH % OPT.saveFreq == 0 then
            save()
        end

        EPOCH = EPOCH + 1
        print("")
    end
end

function save()
    local filename = paths.concat(OPT.save, 'adversarial.net')
    saveAs(filename)
end

-- Save the current models G and D to a file.
-- @param filename The path to the file
function saveAs(filename)
    os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
    if paths.filep(filename) then
      os.execute(string.format("mv %s %s.old", filename, filename))
    end
    print(string.format("<trainer> saving network to %s", filename))
    NN_UTILS.prepareNetworkForSave(MODEL_G)
    NN_UTILS.prepareNetworkForSave(MODEL_D)
    torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, plot_data = PLOT_DATA, epoch = EPOCH, vis_noise_inputs = VIS_NOISE_INPUTS, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD})
end

-- Visualizes the current training status via Display (based on gfx.js) in the browser.
-- It shows:
--   Images generated from random noise (the noise vectors are set once at the start of the
--   training, so the images should end up similar at each epoch)
--   Images that were deemed "good" by D
--   Images that were deemed "bad" by D
--   Original images from the training set (as comparison)
-- @param noiseInputs The noise vectors for the random images.
-- @returns void
function visualizeProgress(noiseInputs)
    -- deactivate dropout
    MODEL_G:evaluate()
    MODEL_D:evaluate()

    -- Generate a synthetic test image as sanity test
    -- This should be deemed very bad by D
    local sanityTestImage = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    sanityTestImage:uniform(0.0, 0.50)
    for i=1,IMG_DIMENSIONS[2] do
        for j=1,IMG_DIMENSIONS[3] do
            if i == j then
                sanityTestImage[1][i][j] = 1.0
            elseif i % 4 == 0 and j % 4 == 0 then
                sanityTestImage[1][i][j] = 0.5
            end
        end
    end

    -- Collect original example images from the training set
    local trainImages = TRAIN_DATA[{{1, 50}, {}, {}, {}}]:clone()

    -- Generate images from G based on the provided noiseInputs
    local rndImages = NN_UTILS.createImagesFromNoise(noiseInputs)

    -- Place the sanity test image and one original image from the training corpus among
    -- the random Images. The first should be deemed bad by D, the latter as good.
    -- Then find good and bad images (according to D) among the randomly generated ones
    -- Note: has to happen before toRgb() as that would change the color space of the images
    local rndImagesClone = rndImages:clone()
    rndImagesClone[rndImagesClone:size(1)-1] = trainImages[1] -- one real face as sanity test
    rndImagesClone[rndImagesClone:size(1)] = sanityTestImage -- synthetic non-face as sanity test
    local goodImages, _ = NN_UTILS.sortImagesByPrediction(rndImagesClone, false, 50)
    local badImages, _ = NN_UTILS.sortImagesByPrediction(rndImagesClone, true, 50)

    if rndImages:ne(rndImages):sum() > 0 then
        print(string.format("[visualizeProgress] Generated images contain NaNs"))
    end

    DISP.image(NN_UTILS.toRgb(rndImages, OPT.colorSpace), {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Generated images (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(goodImages, OPT.colorSpace), {win=OPT.window+2, width=IMG_DIMENSIONS[3]*15, title="Best samples (first is best) (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(badImages, OPT.colorSpace), {win=OPT.window+3, width=IMG_DIMENSIONS[3]*15, title="Worst samples (first is worst) (epoch " .. EPOCH .. ")"})
    DISP.image(NN_UTILS.toRgb(trainImages, OPT.colorSpace), {win=OPT.window+4, width=IMG_DIMENSIONS[3]*15, title="original images from training set"})

    NN_UTILS.saveImagesAsGrid(string.format("%s/images/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(rndImages, OPT.colorSpace), 10, 10, EPOCH)
    NN_UTILS.saveImagesAsGrid(string.format("%s/images_good/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(goodImages, OPT.colorSpace), 7, 7, EPOCH)
    NN_UTILS.saveImagesAsGrid(string.format("%s/images_bad/%d_%05d.png", OPT.save, START_TIME, EPOCH), NN_UTILS.toRgb(badImages, OPT.colorSpace), 7, 7, EPOCH)

    -- reactivate dropout
    MODEL_G:training()
    MODEL_D:training()
end

main()
