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

function models.create_G_facegen(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    model:add(nn.Linear(noiseDim, 128*8*8))
    model:add(nn.View(128, 8, 8))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU(nil, nil, true))

    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, (5-1)/2, (5-1)/2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU(nil, nil, true))

    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_default(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local startHeight = dimensions[2] / 2 / 2 / 2
    local startWidth = dimensions[3] / 2 / 2 / 2

    -- 4x4
    model:add(nn.Linear(noiseDim, 256 * startHeight * startWidth))
    model:add(nn.BatchNormalization(256 * startHeight * startWidth))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(256, startHeight, startWidth))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
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

function models.create_G2(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    local startHeight = dimensions[2] / 2 / 2 / 2
    local startWidth = dimensions[3] / 2 / 2 / 2

    -- 4x4
    model:add(nn.Linear(noiseDim, 512 * startHeight * startWidth))
    model:add(nn.BatchNormalization(512 * startHeight * startWidth))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(512, startHeight, startWidth))

    -- 4x4 -> 8x8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

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

    -- 8x8
    --model:add(nn.CAdd())
    local concat = nn.Concat(2)

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

    model:add(concat)
    model:add(cudnn.SpatialConvolution(32*16, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.PReLU())
    --[[
    model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())
    --]]
    model:add(cudnn.SpatialConvolution(64, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32x32_branched(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    model:add(nn.Linear(noiseDim, 128))
    model:add(cudnn.ReLU(true))

    local conc = nn.Concat(2)
    for i=1,8 do
        local branch = nn.Sequential()

        -- 8x8
        branch:add(nn.Linear(128, 16*8*8))
        branch:add(nn.BatchNormalization(16*8*8))
        branch:add(cudnn.ReLU(true))
        branch:add(nn.View(16, 8, 8))

        -- 8x8 -> 16x16
        branch:add(nn.SpatialUpSamplingNearest(2))
        branch:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        branch:add(nn.SpatialBatchNormalization(16))
        branch:add(cudnn.ReLU(true))

        -- 16x16 -> 32x32
        branch:add(nn.SpatialUpSamplingNearest(2))
        branch:add(cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
        branch:add(nn.SpatialBatchNormalization(16))
        branch:add(cudnn.ReLU(true))

        conc:add(branch)
    end

    model:add(conc)

    model:add(cudnn.SpatialConvolution(16*8, 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(32))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(32, dimensions[1], 1, 1, 1, 1, 0, 0))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling32x48_residual(dimensions, noiseDim, cuda)
    function createReduce(nbInputPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        seq:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        seq:add(nn.SpatialBatchNormalization(nbOutputPlanes))
        seq:add(nn.PReLU())
        return seq
    end

    function createResidual(nbInputPlanes, nbInnerPlanes, nbOutputPlanes)
        local seq = nn.Sequential()
        local inner = nn.Sequential()
        inner:add(cudnn.SpatialConvolution(nbInputPlanes, nbInnerPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbInnerPlanes, 3, 3, 1, 1, 1, 1))
        inner:add(nn.SpatialBatchNormalization(nbInnerPlanes))
        inner:add(cudnn.ReLU(true))
        inner:add(cudnn.SpatialConvolution(nbInnerPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
        inner:add(nn.SpatialBatchNormalization(nbOutputPlanes))
        inner:add(cudnn.ReLU(true))

        local conc = nn.ConcatTable(2)
        conc:add(inner)
        if nbInputPlanes == nbOutputPlanes then
            conc:add(nn.Identity())
        else
            reducer = nn.Sequential()
            reducer:add(cudnn.SpatialConvolution(nbInputPlanes, nbOutputPlanes, 1, 1, 1, 1, 0, 0))
            reducer:add(nn.SpatialBatchNormalization(nbOutputPlanes))
            reducer:add(cudnn.ReLU(true))
            conc:add(reducer)
        end
        seq:add(conc)
        seq:add(nn.CAddTable())
        return seq
    end

    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 4x6
    model:add(nn.Linear(noiseDim, 512*4*6))
    model:add(nn.BatchNormalization(512*4*6))
    model:add(cudnn.ReLU(true))
    model:add(nn.View(512, 4, 6))

    -- 4x6 -> 8x12
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    -- 8x12 -> 16x24
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    -- 16x24 -> 32x48
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))
    model:add(createResidual(128, 32, 128))

    -- decrease to usually 3 dimensions (image channels)
    model:add(cudnn.SpatialConvolution(128, dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    model:add(nn.Sigmoid())

    if cuda then
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        model:cuda()
    end

    model = require('weight-init')(model, 'heuristic')

    return model
end

function models.create_G_decoder_upsampling64x64_residual(dimensions, noiseDim, cuda)
    local model = nn.Sequential()

    if cuda then
        model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 8x8
    model:add(nn.Linear(noiseDim, 512*8*8))
    model:add(nn.BatchNormalization(512*8*8))
    model:add(nn.PReLU())
    model:add(nn.View(512, 8, 8))

    -- 8x8 -> 16x16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.PReLU())

    -- 16x16 -> 32x32
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.PReLU())

    -- 32x32 -> 64x64
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.PReLU())

    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))
    model:add(createResidual(64, 16, 64))

    -- decrease to usually 3 dimensions (image channels)
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
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim, cuda)
    return models.create_G4(dimensions, noiseDim, cuda)
end

-- Creates D.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
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
    --[[
    function createNxN(nbKernelsIn, nbKernelsOut, kernelSize)
        local model = nn.Sequential()

        model:add(nn.SpatialConvolution(nbKernelsIn, nbKernelsOut, 1, kernelSize, 1, 1, 0, (kernelSize-1)/2))
        model:add(nn.PReLU())
        model:add(nn.SpatialConvolution(nbKernelsOut, nbKernelsOut, kernelSize, 1, 1, 1, (kernelSize-1)/2, 0))
        model:add(nn.PReLU())
        return model
    end
    --]]

    function createNxN(nbKernelsIn, nbKernelsOut, kernelSize, dropout)
        local model = nn.Sequential()
        model:add(nn.SpatialConvolution(nbKernelsIn, nbKernelsOut, kernelSize, kernelSize, 1, 1, (kernelSize-1)/2, (kernelSize-1)/2))
        model:add(nn.PReLU())
        if dropout > 0 then
            model:add(nn.SpatialDropout(0.25))
        end
        return model
    end

    function createPooling()
        local model = nn.Concat(2)
        model:add(nn.SpatialAveragePooling(2, 2, 2, 2))
        model:add(nn.SpatialMaxPooling(2, 2))
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

function models.create_D_residual(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(cudnn.ReLU(true))
    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(cudnn.ReLU(true))
    conv:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- 16x16
    conv:add(models.createResidual(64, 32, 128, "ReLU", false))
    conv:add(models.createResidual(128, 32, 128, "ReLU", false))
    conv:add(models.createResidual(128, 32, 128, "ReLU", false))
    conv:add(models.createResidual(128, 32, 128, "ReLU", false))

    conv:add(models.createResidual(128, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(nn.SpatialMaxPooling(2, 2))
    --conv:add(nn.SpatialDropout(0.25))

    -- 8x8
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(models.createResidual(256, 32, 256, "ReLU", false))
    conv:add(nn.SpatialDropout(0.5))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(256*height*width))
    conv:add(nn.Linear(256*height*width, 256))
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

function models.create_R(dimensions, noiseDim, noiseMethod, cuda)
    return models.create_R12(dimensions, noiseDim, noiseMethod, cuda)
end

function models.create_R1(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.Tanh())
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

function models.create_R2(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.LeakyReLU(0.333))
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

function models.create_R3(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.LeakyReLU(0.333))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.LeakyReLU(0.333))
    conv:add(nn.Dropout(0.5))
    conv:add(nn.Linear(512, 512))
    conv:add(nn.LeakyReLU(0.333))
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

function models.create_R4(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    conv:add(nn.Dropout(0.5))
    conv:add(nn.Linear(512, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
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

function models.create_R5(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.Dropout(0.5))
    conv:add(nn.Linear(512, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.Dropout(0.5))
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

function models.create_R6(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.Dropout(0.5))
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

function models.create_R7(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.LeakyReLU(0.333))

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.LeakyReLU(0.333))
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.Tanh())
    --conv:add(nn.Dropout(0.5))
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

function models.create_R8(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.ELU())

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.ELU())
    --conv:add(nn.Dropout(0.5))
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

function models.create_R9(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.ReLU())

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ReLU())
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ReLU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(256))
    conv:add(nn.ReLU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 8x8
    conv:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(512))
    conv:add(nn.ReLU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2 / 2
    local width = dimensions[3] / 2 / 2 / 2

    -- 4x4
    conv:add(nn.View(512*height*width))
    conv:add(nn.Linear(512*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.ReLU())
    --conv:add(nn.Dropout(0.5))
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

function models.create_R10(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.ELU())

    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))

    -- 32x32
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialFractionalMaxPooling(2, 2, 0.75, 0.75))

    -- 16x16
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialFractionalMaxPooling(2, 2, 0.75, 0.75))

    -- 8x8
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialFractionalMaxPooling(2, 2, 0.75, 0.75))

    local height = math.floor(math.floor(math.floor(dimensions[2]*0.75)*0.75)*0.75)
    local width = height

    -- 4x4
    conv:add(nn.View(128*height*width))
    conv:add(nn.Linear(128*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.ELU())
    --conv:add(nn.Dropout(0.5))
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

function models.create_R11(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    -- 32x32
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())

    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(64))
    conv:add(nn.ELU())
    conv:add(nn.SpatialMaxPooling(2, 2))

    -- 16x16
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())

    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.SpatialBatchNormalization(128))
    conv:add(nn.ELU())
    --conv:add(nn.SpatialDropout(0.25))
    conv:add(nn.SpatialMaxPooling(2, 2))

    local height = dimensions[2] / 2 / 2
    local width = dimensions[3] / 2 / 2

    -- 8x8
    conv:add(nn.View(128*height*width))
    conv:add(nn.Linear(128*height*width, 512))
    conv:add(nn.BatchNormalization(512))
    conv:add(nn.ELU())
    --conv:add(nn.Dropout(0.5))
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

function models.create_R12(dimensions, noiseDim, noiseMethod, cuda)
    assert(noiseMethod == "normal" or noiseMethod == "uniform")

    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end

    --[[
    conv:add(nn.SpatialConvolution(dimensions[1], dimensions[1], 3, 3, 1, 1, (3-1)/2, (3-1)/2))
    conv:add(nn.ELU())
    conv:add(nn.Dropout())
    --]]
    --conv:add(nn.Dropout())
    --conv:add(nn.WhiteNoise(0, 0.2))

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
