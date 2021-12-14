const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 1]);
const ys = tf.tensor2d([1, 4, 6, 8, 10, 12, 14, 16, 18, 20], [10, 1]);

async function train() {
    for (let i = 0; i < 100; i++) {
        const response = await model.fit(xs, ys);
        console.log(response.history.loss)
    }
}

train().then(() => {
    model.predict(tf.tensor2d([42], [1, 1])).print()
    model.save('downloads://my-model');
})