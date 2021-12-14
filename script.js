const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// y = 50+2x
const xs = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16, 1]);
const ys = tf.tensor2d([50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80], [16, 1]);

async function train() {
    for (let i = 0; i < 750; i++) {
        const response = await model.fit(xs, ys);
        console.log(response.history.loss)
    }
}

train().then(() => {
    model.predict(tf.tensor2d([26, 27], [2, 1])).print()
    //model.save('downloads://my-model');
})