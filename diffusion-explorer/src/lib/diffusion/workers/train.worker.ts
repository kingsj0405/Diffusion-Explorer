/*
*    Web worker that runs training for a given model. 
*/ 
import * as tf from '@tensorflow/tfjs';
// TODO Fix the wasm implementation
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
setWasmPaths('/tfjs-backend-wasm/');
import '@tensorflow/tfjs-backend-wasm'; // Import the WebGL backend for TensorFlow.js

import { backend, trainingObjectiveToModelClass } from '$lib/settings';

async function loadDataset(path: string) {
    return fetch(path)
        .then(response => response.json())
        .then(data => {
            // Convert the data to a tensor
            const pointsTensor = tf.tensor(data.points);
            return pointsTensor;
        });
}

async function saveModel(model: tf.LayersModel, path: string) {
    // Save the model to IndexedDB
    const modelSaveName = 'indexeddb://' + path.replace(/\s+/g, '_') + '_' + Date.now();
    await model.save(modelSaveName);
    return modelSaveName;
}

// Make a boolean that tracks if the training is stopped
// It is important to be out of the on message function so the value persists across message receives
let trainingStopped = false;

self.onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'train') {
        // Destructure the data
        const { trainingObjective, modelConfig, datasetPath, trainingConfig } = data;
        // Set up tf wasm backend
        if (backend === 'wasm') {
            await tf.setBackend('wasm');
            await tf.ready();
        } else if (backend === 'webgl') {
            await tf.setBackend('webgl');
            await tf.ready();
        }
        // Initialize the empty model 
        const ModelClass = trainingObjectiveToModelClass[trainingObjective];
        const ourModel = new ModelClass(
            modelConfig.dim,
            modelConfig.hidden,
        );
        // Load the dataset
        const pointsTensor = await loadDataset(datasetPath);
        // Run training
        await ourModel.train(
            pointsTensor,
            trainingConfig["epochs"],
            trainingConfig["batchSize"],
            trainingConfig["updateInterval"],
            // Stop training function that handles halting the training
            () => { return trainingStopped; },
            (epoch, intermediateSamples) => {
                // Send the intermediate samples to the main thread
                console.log(epoch, intermediateSamples);
                self.postMessage({ 
                    type: 'epoch_chunk', 
                    epoch: epoch, 
                    intermediateSamples: intermediateSamples,
                });
            }
        )
        const modelSaveName = await saveModel(ourModel.model, trainingObjective)
        console.log("Training worker thread posting result...");
        self.postMessage({ 
            type: 'result', 
            tfModelPath: modelSaveName,
            // allSamples: allSamplesArray,
        });
    } else if (type === 'stop_training') {
        // Figure out how to stop the training
        trainingStopped = true;
    } else {
        console.error('Unknown message type:', type);
    }
};
