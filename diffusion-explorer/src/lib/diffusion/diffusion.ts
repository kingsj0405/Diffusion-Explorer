import * as tf from '@tensorflow/tfjs';
import { Model } from './interfaces';

export class DiffusionModel extends Model {
    readonly T: number;
    readonly betas: tf.Tensor1D;
    readonly alphas: tf.Tensor1D;
    readonly alphasCumprod: tf.Tensor1D;
    readonly alphasCumprodPrev: tf.Tensor1D;
    readonly sqrtAlphasCumprod: tf.Tensor1D;
    readonly sqrtOneMinusAlphasCumprod: tf.Tensor1D;
    readonly sqrtInvAlphasCumprod: tf.Tensor1D;
    readonly sqrtInvAlphasCumprodMinusOne: tf.Tensor1D;
    readonly variance: tf.Tensor1D;
    readonly posteriorCoef1: tf.Tensor1D;
    readonly posteriorCoef2: tf.Tensor1D;
    readonly predictionType: 'epsilon' | 'v';

    constructor(dim = 2, hidden = 128, T = 1000, betaStart = 1e-4, betaEnd = 2e-2, predictionType: 'epsilon' | 'v' = 'epsilon') {
        super(dim, hidden);
        this.T = T;
        this.predictionType = predictionType;
        this.betas = tf.linspace(betaStart, betaEnd, T);
        this.alphas = tf.sub(1, this.betas);
        this.alphasCumprod = tf.cumprod(this.alphas) as tf.Tensor1D;
        this.sqrtAlphasCumprod = tf.sqrt(this.alphasCumprod) as tf.Tensor1D;
        this.sqrtOneMinusAlphasCumprod = tf.sqrt(tf.sub(1, this.alphasCumprod)) as tf.Tensor1D;
        this.sqrtInvAlphasCumprod = tf.sqrt(tf.div(1, this.alphasCumprod)) as tf.Tensor1D;
        this.sqrtInvAlphasCumprodMinusOne = tf.sqrt(tf.sub(tf.div(1, this.alphasCumprod), 1)) as tf.Tensor1D;
        this.alphasCumprodPrev = tf.concat([tf.ones([1]), this.alphasCumprod.slice([0], [T - 1])]) as tf.Tensor1D;
        this.variance = this.betas.mul(tf.sub(1, this.alphasCumprodPrev)).div(tf.sub(1, this.alphasCumprod)).clipByValue(1e-20, 1e20) as tf.Tensor1D;
        
        this.posteriorCoef1 = tf.tidy(() =>
            this.betas.mul(tf.sqrt(this.alphasCumprodPrev)).div(tf.sub(1, this.alphasCumprod))
        ) as tf.Tensor1D;
        this.posteriorCoef2 = tf.tidy(() =>
            tf.sub(1, this.alphasCumprodPrev).mul(tf.sqrt(this.alphas)).div(tf.sub(1, this.alphasCumprod))
        ) as tf.Tensor1D;
    }

    /**
     * Train the diffusion model with denoising score matching or v-prediction
     * @param data tf.Tensor2D of shape [num_samples, dim]
     * @param epochs number of epochs to train the model
     * @param batchSize number of samples to use in each batch
     * @param updateInterval number of epochs to wait before updating the model
     * @param stopTraining function to check if training should stop
     * @param endEpochCallback function to call at the end of each epoch
     * @returns Promise<void>
     */
    async train(
        data: tf.Tensor2D, 
        epochs = 1000, 
        batchSize = 256,
        updateInterval: number = 50,
        stopTraining: () => boolean | Promise<boolean> = () => { return false; },
        endEpochCallback: (epoch: number, intermediateSamples: number[][] | null) => void = () => { },
    ): Promise<void> {
        const N = data.shape[0];
        const optimizer = tf.train.adam(1e-4);
        const mse = (a: tf.Tensor, b: tf.Tensor) => tf.losses.meanSquaredError(a, b);
        const losses: number[] = [];

        for (let epoch = 0; epoch < epochs; ++epoch) {
            for (let i = 0; i < data.shape[0]; i += batchSize) {
                // Create batch data outside of tidy to avoid cleanup issues
                const batchIndices = tf.range(i, Math.min(i + batchSize, N)).toInt();
                const x0 = tf.gather(data, batchIndices) as tf.Tensor2D;
                const noise = tf.randomNormal(x0.shape as [number, number]) as tf.Tensor2D;
                const tInt = tf.randomUniform([x0.shape[0]], 0, this.T, 'int32') as tf.Tensor1D;
                // Add noise to x0
                const x_t = this.addNoise(x0, noise, tInt);
                // Run the optimizer with proper async handling
                let lossValue: number = 0;
                // Create loss computation outside minimize to get the value
                const computeLoss = () => {
                    // Get the model prediction
                    const pred = this.forward(x_t, tInt);
                    // Compute the target based on prediction type
                    let target: tf.Tensor2D;
                    if (this.predictionType === 'epsilon') {
                        target = noise;
                    } else {
                        // v-prediction target: v = alpha_t * epsilon - sigma_t * x0
                        const alpha_t = tf.gather(this.sqrtAlphasCumprod, tInt).expandDims(1);
                        const sigma_t = tf.gather(this.sqrtOneMinusAlphasCumprod, tInt).expandDims(1);
                        target = alpha_t.mul(noise).sub(sigma_t.mul(x0)) as tf.Tensor2D;
                    }
                    // Compute the loss
                    const loss = mse(target, pred);
                    console.log(`Loss: ${loss.dataSync()[0]}`);
                    return loss as tf.Scalar;
                };
                // Get loss value before optimization
                const lossScalar = computeLoss();
                lossValue = lossScalar.dataSync()[0];
                // Run optimization
                optimizer.minimize(computeLoss);
                // Store the loss
                losses.push(lossValue);
            }
            // Run intermediate sampling
            let intermediateSamples = null;
            if (epoch % updateInterval === 0) {
                // Sample from the model
                const allTimeSamples = this.sample(
                    500, // number of samples
                    100 // number of steps
                ); // shape [num_total_steps, num_samples, dim]
                // Pull out the last time step
                const lastTimeStep = allTimeSamples.gather(allTimeSamples.shape[0] - 1, 0);
                intermediateSamples = (lastTimeStep as unknown as tf.Tensor2D).arraySync() as number[][];
            }
            // Run the end epoch callback
            // TODO: add the loss
            endEpochCallback(epoch, intermediateSamples);
            // Yield control to the worker event loop to handle stop events
            await tf.nextFrame();
            // Check if the training should continue
            if (stopTraining()) {
                console.log("Training stopped by user.");
                break;
            }
        }
    }

    private addNoise(x0: tf.Tensor2D, noise: tf.Tensor2D, tInt: tf.Tensor1D): tf.Tensor2D {
        return tf.tidy(() => {
            const s1 = tf.gather(this.sqrtAlphasCumprod, tInt).expandDims(1);
            const s2 = tf.gather(this.sqrtOneMinusAlphasCumprod, tInt).expandDims(1);
            return x0.mul(s1).add(noise.mul(s2)) as tf.Tensor2D;
        });
    }

    forward(x_t: tf.Tensor2D, t: tf.Tensor1D | tf.Tensor2D): tf.Tensor2D {
        return tf.tidy(() => {
            const t_expanded = t.reshape([x_t.shape[0], 1]); // Use very simple time conditioning, no sinusoidal embedding
            const t_scaled = t_expanded.div(this.T);
            const input = tf.concat([x_t, t_scaled], 1); // shape [batch, dim+1]

            return this.model.predict(input) as tf.Tensor2D;
        });
    }

    step(x_t: tf.Tensor2D, t_start: tf.Tensor1D | tf.Tensor2D): tf.Tensor2D {
        return tf.tidy(() => {
            // Reconstruct the original sample
            const tInt = (t_start.rank === 2 ? t_start.squeeze() : t_start) as tf.Tensor1D;
            const pred = this.forward(x_t, tInt);
            
            let eps_hat: tf.Tensor2D;
            if (this.predictionType === 'epsilon') {
                eps_hat = pred;
            } else {
                // Convert v-prediction to epsilon: epsilon = alpha_t * v + sigma_t * x_t
                const alpha_t = tf.gather(this.sqrtAlphasCumprod, tInt).expandDims(1);
                const sigma_t = tf.gather(this.sqrtOneMinusAlphasCumprod, tInt).expandDims(1);
                eps_hat = alpha_t.mul(pred).add(sigma_t.mul(x_t)) as tf.Tensor2D;
            }
            
            // Reconstruct the original sample using epsilon
            const s1 = tf.gather(this.sqrtInvAlphasCumprod, tInt).expandDims(1);
            const s2 = tf.gather(this.sqrtInvAlphasCumprodMinusOne, tInt).expandDims(1);
            const pred_original_sample = x_t.mul(s1).sub(eps_hat.mul(s2));
            // Predict the previous sample
            const c1 = tf.gather(this.posteriorCoef1, tInt).expandDims(1);
            const c2 = tf.gather(this.posteriorCoef2, tInt).expandDims(1);
            const pred_prev_sample = x_t.mul(c2).add(pred_original_sample.mul(c1));
            // Add noise back to the sample
            const noise = tf.randomNormal(x_t.shape as [number, number]);
            const varTerm = tf.gather(this.variance, tInt).sqrt().expandDims(1).mul(noise);
            const isZero = tInt.equal(tf.scalar(0, 'int32')).expandDims(1);
            return pred_prev_sample.add(varTerm.mul(tf.cast(isZero.logicalNot(), 'float32')));
        });
    }

    sample(num_samples: number, num_total_steps: number = this.T): tf.Tensor3D {
        return tf.tidy(() => {
            // Draw some initial samples from the source distribution
            let x: tf.Tensor2D = tf.randomNormal([num_samples, this.dim]);
            const traj: tf.Tensor2D[] = [];
            const steps = [...Array(num_total_steps).keys()].reverse();
            // Iterate through the timesteps backwards
            for (const t of steps) {
                const tInt = tf.fill([num_samples], t, 'int32') as tf.Tensor1D;
                x = this.step(x, tInt);
                traj.push(x);
            }
            return tf.stack(traj) as tf.Tensor3D;
        });
    }

    sample_from_initial_points(initial_points: tf.Tensor2D, num_total_steps: number = this.T): tf.Tensor3D {
        return tf.tidy(() => {
            let x = initial_points;
            const traj: tf.Tensor2D[] = [];
            const steps = [...Array(num_total_steps).keys()].reverse();
            for (const t of steps) {
                traj.push(x);
                const tInt = tf.fill([x.shape[0]], t, 'int32') as tf.Tensor1D;
                x = this.step(x, tInt);
            }
            traj.push(x);
            return tf.stack(traj) as tf.Tensor3D;
        });
    }
}