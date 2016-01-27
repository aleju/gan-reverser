require 'torch'

local nn_utils = {}

function nn_utils.forwardBatched(model, input, batchSize)
    local N
    if input.size then
        N = input:size(1)
    else
        N = #input
    end

    local output
    local nBatches = math.ceil(N/batchSize)
    for i=1,nBatches do
        local batchStart = 1 + (i-1) * batchSize
        local batchEnd = math.min(i*batchSize, N)
        local forwarded = model:forward(input[{{batchStart, batchEnd}}]):clone()
        if output == nil then
            local sizes = forwarded:size()
            sizes[1] = N
            output = torch.Tensor():resize(sizes)
        end

        for j=1,forwarded:size(1) do
            --print("Writing forwarded@", j, " to output@", batchStart+j-1)
            output[batchStart+j-1] = forwarded[j]
        end
        --output[{{batchStart, batchEnd}, {}, {}, {}}] = forwarded
    end

    return output
end

-- Creates a tensor of N vectors, each of dimension OPT.noiseDim with random values
-- between -1 and +1.
-- @param N Number of vectors to generate
-- @returns Tensor of shape (N, OPT.noiseDim)
function nn_utils.createNoiseInputs(N, noiseDim, method)
    noiseDim = noiseDim or OPT.noiseDim
    local method = method or OPT.noiseMethod
    local noiseInputs = torch.Tensor(N, noiseDim)
    if method == "uniform" then
        noiseInputs:uniform(-1.0, 1.0)
    elseif method == "normal" then
        noiseInputs:normal(0.0, 1.0)
    else
        error(string.format("Unknown noise method '%s'", method))
    end
    return noiseInputs
end

-- Feeds noise vectors into G or AE+G and returns the result.
-- @param noiseInputs Tensor from createNoiseInputs()
-- @param outputAsList Whether to return the images as one list or as a tensor.
-- @returns Either list of images (as returned by G/AE) or tensor of images
function nn_utils.createImagesFromNoise(noiseInputs, outputAsList)
    local images
    local N = noiseInputs:size(1)
    local nBatches = math.ceil(N/OPT.batchSize)
    for i=1,nBatches do
        local batchStart = 1 + (i-1)*OPT.batchSize
        local batchEnd = math.min(i*OPT.batchSize, N)
        local generated = MODEL_G:forward(noiseInputs[{{batchStart, batchEnd}}]):clone()
        if images == nil then
            local img = generated[1]
            images = torch.Tensor(N, img:size(1), img:size(2), img:size(3))
        end
        images[{{batchStart, batchEnd}, {}, {}, {}}] = generated
    end

    if outputAsList then
        local imagesList = {}
        for i=1, images:size(1) do
            imagesList[#imagesList+1] = images[i]:float()
        end
        return imagesList
    else
        return images
    end
end

-- Creates new random images with G or AE+G.
-- @param N Number of images to create.
-- @param outputAsList Whether to return the images as one list or as a tensor.
-- @returns Either list of images (as returned by G/AE) or tensor of images
function nn_utils.createImages(N, outputAsList)
    return nn_utils.createImagesFromNoise(nn_utils.createNoiseInputs(N), outputAsList)
end

-- Sorts images based on D's certainty that they are fake/real.
-- Descending order starts at y=1 (Y_NOT_GENERATOR) and ends with y=0 (Y_GENERATOR).
-- Therefore, in case of descending order, images for which D is very certain that they are real
-- come first and images that seem to be fake (according to D) come last.
-- @param images Tensor of the images to sort.
-- @param ascending If true then images that seem most fake to D are placed at the start of the list.
--                  Otherwise the list starts with probably real images.
-- @param nbMaxOut Sets how many images may be returned max (cant be more images than provided).
-- @return Tuple (list of images, list of predictions between 0.0 and 1.0)
--                                where 1.0 means "probably real"
function nn_utils.sortImagesByPrediction(images, ascending, nbMaxOut)
    local predictions = torch.Tensor(images:size(1), 1)
    local nBatches = math.ceil(images:size(1)/OPT.batchSize)
    for i=1,nBatches do
        local batchStart = 1 + (i-1)*OPT.batchSize
        local batchEnd = math.min(i*OPT.batchSize, images:size(1))
        predictions[{{batchStart, batchEnd}, {1}}] = MODEL_D:forward(images[{{batchStart, batchEnd}, {}, {}, {}}]):clone()
    end

    local imagesWithPreds = {}
    for i=1,images:size(1) do
        table.insert(imagesWithPreds, {images[i], predictions[i][1]})
    end

    if ascending then
        table.sort(imagesWithPreds, function (a,b) return a[2] < b[2] end)
    else
        table.sort(imagesWithPreds, function (a,b) return a[2] > b[2] end)
    end

    resultImages = {}
    resultPredictions = {}
    for i=1,math.min(nbMaxOut,#imagesWithPreds) do
        resultImages[i] = imagesWithPreds[i][1]
        resultPredictions[i] = imagesWithPreds[i][2]
    end

    return resultImages, resultPredictions
end



function nn_utils.switchColorSpace(images, from, to)
    images = nn_utils.toRgb(images, from)
    images = nn_utils.rgbToColorSpace(images, to)
    return images
end

function nn_utils.toRgb(images, from)
    local images = nn_utils.toImageTensor(images)
    if from == "rgb" then
        return images
    elseif from == "y" then
        return torch.repeatTensor(images, 1, 3, 1, 1)
    elseif from == "hsl" then
        local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4))
        for i=1,images:size(1) do
            out[i] = image.hsl2rgb(images[i])
        end
        return out
    elseif from == "yuv" then
        local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4))
        for i=1,images:size(1) do
            out[i] = image.yuv2rgb(images[i])
        end
        return out
    else
        error("[WARNING] unknown color space <from>: '" .. from .. "'")
    end
end

function nn_utils.rgbToColorSpace(images, colorSpace)
    if colorSpace == "rgb" then
        return images
    else
        if colorSpace == "y" then
            local out = torch.Tensor(images:size(1), 1, images:size(3), images:size(4))
            for i=1,images:size(1) do
                out[i] = nn_utils.rgb2y(images[i])
            end
            return out
        elseif colorSpace == "hsl" then
            local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4))
            for i=1,images:size(1) do
                out[i] = image.rgb2hsl(images[i])
            end
            return out
        elseif colorSpace == "yuv" then
            local out = torch.Tensor(images:size(1), 3, images:size(3), images:size(4))
            for i=1,images:size(1) do
                out[i] = image.rgb2yuv(images[i])
            end
            return out
        else
            print("[WARNING] unknown color space in rgbToColorSpace: '" .. colorSpace .. "'")
        end
    end
end

-- convert rgb to grayscale by averaging channel intensities
-- https://gist.github.com/jkrish/29ca7302e98554dd0fcb
function nn_utils.rgb2y(im, threeChannels)
    -- Image.rgb2y uses a different weight mixture
    local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
    if dim ~= 3 then
        print('<error> expected 3 channels')
        return im
    end

    -- a cool application of tensor:select
    local r = im:select(1, 1)
    local g = im:select(1, 2)
    local b = im:select(1, 3)

    local z = torch.Tensor(1, w, h):zero()

    -- z = z + 0.21r
    z = z:add(0.21, r)
    z = z:add(0.72, g)
    z = z:add(0.07, b)

    if threeChannels == true then
        z = torch.repeatTensor(z, 3, 1, 1)
    end

    return z
end

-- Convert a list (table) of images to a Tensor.
-- If the parameter is already a tensor, it will be returned unchanged.
-- @param imageList A non-empty list/table or tensor of images (each being a tensor).
-- @returns A tensor of shape (N, channels, height, width)
function nn_utils.toImageTensor(imageList, forceChannel)
    if imageList.size ~= nil then
        if not forceChannel or (#imageList:size() == 3) then
            return imageList
        else
            -- forceChannel activated and images lack channel dimension
            -- add it
            local tens = torch.Tensor(imageList:size(1), 1, imageList:size(2), imageList:size(3))
            for i=1,imageList:size(1) do
                tens[i][1] = imageList[i]
            end
            return tens
        end
    else
        if forceChannel == nil then
            forceChannel = false
        end

        local hasChannel = (#imageList[1]:size() == 3)

        local tens
        if hasChannel then
            tens = torch.Tensor(#imageList, imageList[1]:size(1), imageList[1]:size(2), imageList[1]:size(3))
        elseif not hasChannel and forceChannel then
            tens = torch.Tensor(#imageList, 1, imageList[1]:size(1), imageList[1]:size(2))
        else
            tens = torch.Tensor(#imageList, imageList[1]:size(1), imageList[1]:size(2))
        end

        for i=1,#imageList do
            if (not hasChannel and forceChannel) then
                tens[i][1] = imageList[i]
            else
                tens[i] = imageList[i]
            end
        end
        return tens
    end
end

function nn_utils.toImageList(imageTensor, forceChannel)
    local tens = nn_utils.toImageTensor(imageTensor, forceChannel)
    local lst = {}
    for i=1,tens:size(1) do
        table.insert(lst, tens[i])
    end
    return lst
end

-- Normalize given images, currently to range -1.0 (black) to +1.0 (white), assuming that
-- the input images are normalized to range 0.0 (black) to +1.0 (white).
-- @param data Tensor of images
-- @param mean_ Currently ignored.
-- @param std_ Currently ignored.
-- @return (mean, std), both currently always 0.5 dummy values
function nn_utils.normalize(data, mean_, std_)
    -- Code to normalize to zero-mean and unit-variance.
    --[[
    local mean = mean_ or data:mean(1)
    local std = std_ or data:std(1, true)
    local eps = 1e-7
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        data[i]:add(-1, mean)
        data[i]:cdiv(std + eps)
    end

    return mean, std
    --]]

    -- Code to normalize to range -1.0 to +1.0, where -1.0 is black and 1.0 is the maximum
    -- value in this image.
    --[[
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        local m = torch.max(data[i])
        data[i]:div(m * 0.5)
        data[i]:add(-1.0)
        data[i] = torch.clamp(data[i], -1.0, 1.0)
    end
    --]]

    -- Normalize to range -1.0 to +1.0, where -1.0 is black and +1.0 is white.
    local N
    if data.size ~= nil then
        N = data:size(1)
    else
        N = #data
    end

    for i=1,N do
        data[i]:mul(2)
        data[i]:add(-1.0)
        data[i] = torch.clamp(data[i], -1.0, 1.0)
    end

    -- Dummy return values
    return 0.5, 0.5
end

-- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
function nn_utils.zeroDataSize(data)
    if type(data) == 'table' then
        for i = 1, #data do
            data[i] = nn_utils.zeroDataSize(data[i])
        end
    elseif type(data) == 'userdata' then
        data = torch.Tensor():typeAs(data)
    end
    return data
end

-- from https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
-- Resize the output, gradInput, etc temporary tensors to zero (so that the on disk size is smaller)
function nn_utils.prepareNetworkForSave(node)
    if node.output ~= nil then
        node.output = nn_utils.zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = nn_utils.zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = nn_utils.zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                nn_utils.prepareNetworkForSave(child)
            end
        end
    end
    collectgarbage()
end

function nn_utils.getNumberOfParameters(net)
    local nparams = 0
    local dModules = net:listModules()
    for i=1,#dModules do
        if dModules[i].weight ~= nil then
            nparams = nparams + dModules[i].weight:nElement()
        end
    end
    return nparams
end

-- Contains the pixels necessary to draw digits 0 to 9
CHAR_TENSORS = {}
CHAR_TENSORS[0] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {1, 1, 1}})
CHAR_TENSORS[1] = torch.Tensor({{0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1}})
CHAR_TENSORS[2] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1}})
CHAR_TENSORS[3] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {0, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}})
CHAR_TENSORS[4] = torch.Tensor({{1, 0, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {0, 0, 1},
                                {0, 0, 1}})
CHAR_TENSORS[5] = torch.Tensor({{1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}})
CHAR_TENSORS[6] = torch.Tensor({{1, 1, 1},
                                {1, 0, 0},
                                {1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1}})
CHAR_TENSORS[7] = torch.Tensor({{1, 1, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1},
                                {0, 0, 1}})
CHAR_TENSORS[8] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1}})
CHAR_TENSORS[9] = torch.Tensor({{1, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1},
                                {0, 0, 1},
                                {1, 1, 1}})

-- Converts a list of images to a grid of images that can be saved easily.
-- It will also place the epoch number at the bottom of the image.
-- At least parts of this function probably should have been a simple call
-- to image.toDisplayTensor().
-- @param images Tensor of image tensors
-- @param height Height of the grid
-- @param width Width of the grid
-- @param epoch The epoch number to draw at the bottom of the grid
-- @returns tensor
function nn_utils.imagesToGridTensor(images, height, width, epoch)
    local imgChannels = images:size(2)
    local imgHeightPx = IMG_DIMENSIONS[2]
    local imgWidthPx = IMG_DIMENSIONS[3]
    local heightPx = height * imgHeightPx + (1 + 5 + 1)
    local widthPx = width * imgWidthPx
    local grid = torch.Tensor(imgChannels, heightPx, widthPx)
    grid:zero()

    -- add images to grid, one by one
    local yGridPos = 1
    local xGridPos = 1
    for i=1,math.min(images:size(1), height*width) do
        -- set pixels of image
        local yStart = 1 + ((yGridPos-1) * imgHeightPx)
        local yEnd = yStart + imgHeightPx - 1
        local xStart = 1 + ((xGridPos-1) * imgWidthPx)
        local xEnd = xStart + imgWidthPx - 1
        grid[{{1,imgChannels}, {yStart,yEnd}, {xStart,xEnd}}] = images[i]:float()

        -- move to next position in grid
        xGridPos = xGridPos + 1
        if xGridPos > width then
            xGridPos = 1
            yGridPos = yGridPos + 1
        end
    end

    -- add the epoch at the bottom of the image
    local epochStr = tostring(epoch)
    local pos = 1
    for i=epochStr:len(),1,-1 do
        local c = tonumber(epochStr:sub(i,i))
        for channel=1,imgChannels do
            local yStart = heightPx - 1 - 5 -- constant for all
            local yEnd = yStart + 5 - 1 -- constant for all
            local xStart = widthPx - 1 - pos*5 - pos
            local xEnd = xStart + 3 - 1

            grid[{{channel}, {yStart, yEnd}, {xStart, xEnd}}] = CHAR_TENSORS[c]
        end
        pos = pos + 1
    end

    return grid
end

-- Saves the list of image to the provided filepath (as a grid image).
-- @param filepath Save the grid image to that filepath
-- @param images List of image tensors
-- @param height Height of the grid
-- @param width Width of the grid
-- @param epoch The epoch number to draw at the bottom of the grid
-- @returns tensor
function nn_utils.saveImagesAsGrid(filepath, images, height, width, epoch)
    local grid = nn_utils.imagesToGridTensor(images, height, width, epoch)
    os.execute(string.format("mkdir -p %s", sys.dirname(filepath)))
    image.save(filepath, grid)
end

return nn_utils
