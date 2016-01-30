require 'torch'
require 'nn'
require 'dpnn'
require 'cudnn'

local models = {}

function models.createResidual(nbInputPlanes, nbInnerPlanes, nbOutputPlanes, activation, bn)
    if activation == nil or activation == "ReLU" then
        activation = function () return cudnn.ReLU(true) end
    elseif activation == "PReLU" then
        activation = function () return nn.PReLU() end
    elseif activation == "LeakyReLU" then
        activation = function () return nn.LeakyReLU(0.333) end
    else
        error(string.format("Unknown activation '%s'", activation))
    end

    if bn == nil then bn = true end
    assert(bn == true or bn == false)

    local seq = nn.Sequential()
    local inner = nn.Sequential()
    if nbInputPlanes ~= nbInnerPlanes then
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 1, 1, 1, 1, 0, 0))
        if bn then inner:add(nn.SpatialBatchNormalization(nbInnerPlanes)) end
        inner:add(activation())
    end
    inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
    if bn then inner:add(nn.SpatialBatchNormalization(nbInnerPlanes)) end
    inner:add(activation())
    inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
    if bn then inner:add(nn.SpatialBatchNormalization(nbInnerPlanes)) end
    inner:add(activation())
    if nbInnerPlanes ~= nbOutputPlanes then
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        if bn then inner:add(nn.SpatialBatchNormalization(nbOutputPlanes)) end
        inner:add(activation())
    end

    local conc = nn.ConcatTable(2)
    conc:add(inner)
    if nbInputPlanes == nbOutputPlanes then
        conc:add(nn.Identity())
    else
        reducer = nn.Sequential()
        reducer:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        if bn then reducer:add(nn.SpatialBatchNormalization(nbOutputPlanes)) end
        reducer:add(activation())
        conc:add(reducer)
    end
    seq:add(conc)
    seq:add(nn.CAddTable())
    return seq
end

function models.create_G_encoder(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local startHeight = dimensions[2]
    local startWidth = dimensions[3]

    -- 32x32 -> 16x16
    model:add(cudnn.SpatialConvolution(dimensions[1], 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(16))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x16 -> 8x8
    model:add(cudnn.SpatialConvolution(16, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8 -> 4x4
    model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2))

    local height = startHeight/2/2/2
    local width = startWidth/2/2/2
    model:add(nn.View(64*height*width))
    model:add(nn.Linear(64*height*width, 512))
    model:add(nn.BatchNormalization(512))
    model:add(cudnn.ReLU(true))
    model:add(nn.Linear(512, noiseDim))
    model:add(nn.Tanh())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G3(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local startHeight = dimensions[2] / 2 / 2
    local startWidth = dimensions[3] / 2 / 2

    -- 8x8
    model:add(nn.Linear(noiseDim, 512 * startHeight * startWidth))
    model:add(nn.BatchNormalization(512 * startHeight * startWidth))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(512, startHeight, startWidth))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G4(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local startHeight = dimensions[2] / 2 / 2
    local startWidth = dimensions[3] / 2 / 2

    local concat = nn.Concat(2)

    -- 32 branches of [1x16 -> 16x16x16 -> 16x32x32]
    for i=1,32 do
        local seq = nn.Sequential()
        seq:add(nn.Linear(noiseDim, 16))
        seq:add(nn.PReLU())
        seq:add(nn.Linear(16, 16*16*16))
        seq:add(nn.BatchNormalization(16*16*16))
        seq:add(nn.PReLU())
        seq:add(nn.Reshape(16, 16, 16))

        seq:add(nn.SpatialUpSamplingNearest(2))
        seq:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        seq:add(nn.SpatialBatchNormalization(16))
        seq:add(nn.PReLU())
        concat:add(seq)
    end

    -- Merge 32 branches to (32*16)x32x32 = 512x32x32
    model:add(concat)

    -- 512x32x32 -> 64x32x32
    model:add(cudnn.SpatialConvolution(32*16, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.PReLU())

    -- 64x32x32 -> 3x32x32
    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @param cuda Whether to create the model in CUDA/GPU mode.
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim, cuda)
    return models.create_G3(dimensions, noiseDim, cuda)
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param cuda Whether to create the model in CUDA/GPU mode.
-- @returns nn.Sequential
function models.create_D(dimensions, cuda)
    return models.create_D2(dimensions, cuda)
end

function models.create_D_default(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --conv:add(cudnn.ReLU(true))
    conv:add(nn.PReLU())
    --conv:add(nn.SpatialDropout(0.25))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --conv:add(cudnn.ReLU(true))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --conv:add(cudnn.ReLU(true))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --conv:add(cudnn.ReLU(true))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    --conv:add(cudnn.ReLU(true))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout(0.5))
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D2(dimensions, cuda)
    function createNxN(nbKernelsIn, nbKernelsOut, kernelSize, dropout)
        local model = nn.Sequential()
        model:add(nn.SpatialConvolution(nbKernelsIn, nbKernelsOut, kernelSize, kernelSize, 1, 1, (kernelSize-1)/2, (kernelSize-1)/2))
        model:add(nn.PReLU())
        if dropout > 0 then
            model:add(nn.SpatialDropout(0.25))
        end
        return model
    end

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(createNxN(dimensions[1], 128, 3, 0))
    conv:add(createNxN(128, 128, 3, 0.2))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local concat = nn.Concat(2)
    local left = nn.Sequential()
    local right = nn.Sequential()

    left:add(createNxN(128, 64, 5, 0.2))
    left:add(nn.SpatialMaxPooling(2, 2))
    left:add(nn.View(64 * (dimensions[2]/2/2) * (dimensions[3]/2/2)))
    left:add(nn.Linear(64 * (dimensions[2]/2/2) * (dimensions[3]/2/2), 512))
    left:add(nn.PReLU())
    left:add(nn.Dropout(0.25))

    right:add(createNxN(128, 128, 3, 0.2))
    right:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    right:add(createNxN(128, 256, 3, 0.2))
    right:add(createNxN(256, 256, 3, 0.2))
    right:add(nn.SpatialMaxPooling(2, 2))

    -- 4x4
    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2
    right:add(nn.View(256*height*width))
    right:add(nn.Linear(256*height*width, 512))
    right:add(nn.PReLU())

    concat:add(left)
    concat:add(right)
    conv:add(concat)

    conv:add(nn.Linear(512 + 512, 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout(0.25))
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_D_facegen(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.SpatialDropout(0.2))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    conv:add(nn.View(512 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(512 * 0.25 * 0.25 * 0.25 * 0.25 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 512))
    conv:add(nn.PReLU(nil, nil, true))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

function models.create_R(dimensions, noiseDim, noiseMethod, fixer, cuda)
    return models.create_R_default(dimensions, noiseDim, noiseMethod, fixer, cuda)
end

function models.create_R_default(dimensions, noiseDim, noiseMethod, fixer, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- for the error fixer, add a dropout layer between input (image) and the
    -- first layer
    if fixer then
        -- make the dropout layer always activate (even outside training)
        -- if its deactivatable, the generated images tend to look bad
        local drop = nn.Dropout(0.5, true)
        drop:training()
        drop.evaluate = function() end
        conv:add(drop)
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    conv:add(nn.Dropout())

    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    conv:add(nn.Dropout())

    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout())

    -- 16x16
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    conv:add(nn.Dropout())

    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    conv:add(nn.Dropout())

    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2
    local width = dimensions[3] / 2 / 2

    -- 8x8
    conv:add(nn.View(128*height*width))
    conv:add(nn.Linear(128*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.ELU())
    conv:add(nn.Dropout(0.5))
    conv:add(nn.Linear(512, noiseDim))
    if noiseMethod ~= "normal" then
        conv:add(nn.Tanh())
    end

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

return models
