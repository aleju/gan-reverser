require 'torch'
require 'optim'
require 'pl'
require 'image'

local adversarial = {}

function adversarial.clamp(gradParameters, clampValue)
    if clampValue ~= 0 then
        gradParameters:clamp((-1)*clampValue, clampValue)
    end
end

function adversarial.l1(parameters, gradParameters, lossValue, l1weight)
    if l1weight ~= 0 then
        lossValue = lossValue + l1weight * torch.norm(parameters, 1)
        gradParameters:add(torch.sign(parameters):mul(l1Weight))
    end
    return lossValue
end

function adversarial.l2(parameters, gradParameters, lossValue, l2weight)
    if l2weight ~= 0 then
        lossValue = lossValue + l2weight * torch.norm(parameters, 2)^2/2
        gradParameters:add(parameters:clone():mul(l2weight))
    end
    return lossValue
end

function adversarial.showBatch(images, windowId, title, width)
    title = title or string.format("Batch at Epoch %d", EPOCH)
    width = width or IMG_DIMENSIONS[3] * 15
    DISP.image(NN_UTILS.toRgb(images, OPT.colorSpace), {win=windowId, width=width, title=title})
end

-- main training function
function adversarial.train(trainData)
    EPOCH = EPOCH or 1
    local batchSize = OPT.batchSize
    local noiseDim = OPT.noiseDim
    local batchesPerEpoch = OPT.N_epoch
    if batchesPerEpoch <= 0 then batchesPerEpoch = 100 end

    local exampleForDIdx = 1 -- Which example from the training set to chose next to train D.
    local nbExamplesD = 0
    local nbExamplesG = 0

    -- do one epoch
    -- While this function is structured like one that picks example batches in consecutive order,
    -- in reality the examples (per batch) will be picked randomly
    print(string.format("<trainer> Epoch #%d [batchSize = %d]", EPOCH, batchSize))
    for batchIdx=1,batchesPerEpoch do
        local inputs = torch.Tensor(batchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(batchSize)
        local noiseInputs --= torch.Tensor(batchSize, noiseDim)

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of D
        local fevalD = function(x)
            collectgarbage()

            if x ~= PARAMETERS_D then -- get new parameters
                PARAMETERS_D:copy(x)
            end

            GRAD_PARAMETERS_D:zero() -- reset gradients

            adversarial.showBatch(inputs, OPT.window+10, string.format("Training batch for D (BatchIdx=%d, Epoch %d)", batchIdx, EPOCH))

            --  forward pass
            local outputs = MODEL_D:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass
            local df_do = CRITERION:backward(outputs, targets)
            MODEL_D:backward(inputs, df_do)

            f = adversarial.l1(PARAMETERS_D, GRAD_PARAMETERS_D, f, OPT.D_L1)
            f = adversarial.l2(PARAMETERS_D, GRAD_PARAMETERS_D, f, OPT.D_L2)
            adversarial.clamp(GRAD_PARAMETERS_D, OPT.D_clamp)

            -- update confusion (add 1 since targets are binary)
            for i=1,outputs:size(1) do
                local c
                if outputs[i][1] > 0.5 then c = 1 else c = 0 end
                CONFUSION:add(c+1, targets[i]+1)
            end

            return f,GRAD_PARAMETERS_D
        end

        ----------------------------------------------------------------------
        -- create closure to evaluate f(X) and df/dX of generator
        local fevalG_on_D = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G then -- get new parameters
                PARAMETERS_G:copy(x)
            end

            GRAD_PARAMETERS_G:zero() -- reset gradients

            -- forward pass
            local samples = MODEL_G:forward(noiseInputs)
            local outputs = MODEL_D:forward(samples)
            local f = CRITERION:forward(outputs, targets)
            adversarial.showBatch(samples, OPT.window+11, string.format("Training batch for G, rated by D (BatchIdx=%d, Epoch %d)", batchIdx, EPOCH))

            --  backward pass
            local df_samples = CRITERION:backward(outputs, targets)
            MODEL_D:backward(samples, df_samples)
            local df_do = MODEL_D.modules[1].gradInput
            --print(noiseInputs:size())
            --print(df_do:size())
            MODEL_G:backward(noiseInputs, df_do)

            f = adversarial.l1(PARAMETERS_G, GRAD_PARAMETERS_G, f, OPT.G_L1)
            f = adversarial.l2(PARAMETERS_G, GRAD_PARAMETERS_G, f, OPT.G_L2)
            adversarial.clamp(GRAD_PARAMETERS_G, OPT.G_clamp)

            return f,GRAD_PARAMETERS_G
        end
        ------------------- end of eval functions ---------------------------

        ----------------------------------------------------------------------
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        for k=1,OPT.D_iterations do
            -- (1.1) Real data
            local inputIdx = 1
            for i=1,batchSize/2 do
                --local randomIdx = math.random(trainData:size())
                --local exampleIdx = (batchIdx-1) * (batchSize/2) + i
                inputs[inputIdx] = trainData[exampleForDIdx]:clone()
                targets[inputIdx] = Y_NOT_GENERATOR
                inputIdx = inputIdx + 1
                exampleForDIdx = exampleForDIdx + 1
            end

            -- (1.2) Sampled data
            local samples = NN_UTILS.createImages(batchSize/2, false)
            for i=1,batchSize/2 do
                inputs[inputIdx] = samples[i]:clone()
                targets[inputIdx] = Y_GENERATOR
                inputIdx = inputIdx + 1
            end

            if OPT.D_optmethod == "sgd" then
                optim.sgd(fevalD, PARAMETERS_D, OPTSTATE.sgd.D)
            elseif OPT.D_optmethod == "adagrad" then
                optim.adagrad(fevalD, PARAMETERS_D, OPTSTATE.adagrad.D)
            elseif OPT.D_optmethod == "adadelta" then
                optim.adadelta(fevalD, PARAMETERS_D, OPTSTATE.adadelta.D)
            elseif OPT.D_optmethod == "adamax" then
                optim.adamax(fevalD, PARAMETERS_D, OPTSTATE.adamax.D)
            elseif OPT.D_optmethod == "adam" then
                optim.adam(fevalD, PARAMETERS_D, OPTSTATE.adam.D)
            elseif OPT.D_optmethod == "rmsprop" then
                optim.rmsprop(fevalD, PARAMETERS_D, OPTSTATE.rmsprop.D)
            else
                error(string.format("Unknown optimizer method '%s' chosen for D.", OPT.D_optmethod))
            end

            nbExamplesD = nbExamplesD + inputs:size(1)
        end

        ----------------------------------------------------------------------
        -- (2) Update G network: maximize log(D(G(z)))
        for k=1,OPT.G_iterations do
            --noiseInputs = NN_UTILS.createNoiseInputs(noiseInputs:size(1))
            --noiseInputs:uniform(-1, 1)
            noiseInputs = NN_UTILS.createNoiseInputs(batchSize)
            targets:fill(Y_NOT_GENERATOR)

            if OPT.G_optmethod == "sgd" then
                optim.sgd(fevalG_on_D, PARAMETERS_G, OPTSTATE.sgd.G)
            elseif OPT.G_optmethod == "adagrad" then
                optim.adagrad(fevalG_on_D, PARAMETERS_G, OPTSTATE.adagrad.G)
            elseif OPT.G_optmethod == "adadelta" then
                optim.adadelta(fevalG_on_D, PARAMETERS_G, OPTSTATE.adadelta.G)
            elseif OPT.G_optmethod == "adamax" then
                optim.adamax(fevalG_on_D, PARAMETERS_G, OPTSTATE.adamax.G)
            elseif OPT.G_optmethod == "adam" then
                optim.adam(fevalG_on_D, PARAMETERS_G, OPTSTATE.adam.G)
            elseif OPT.G_optmethod == "rmsprop" then
                optim.rmsprop(fevalG_on_D, PARAMETERS_G, OPTSTATE.rmsprop.G)
            else
                error(string.format("Unknown optimizer method '%s' chosen for G.", OPT.G_optmethod))
            end

            nbExamplesG = nbExamplesG + noiseInputs:size(1)
        end

        -- display progress
        xlua.progress(batchIdx, batchesPerEpoch)
    end

    print(string.format("Trained G on: %d | Trained D on: %d", nbExamplesG, nbExamplesD))

    -- print confusion matrix
    print(CONFUSION)
    local tV = CONFUSION.totalValid
    CONFUSION:zero()

    return tV
end

return adversarial
