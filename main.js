import * as tf from '@tensorflow/tfjs';
const btnCalcularPeso = document.getElementById('valorAltura');
const contenedorResultado = document.getElementById('resultado');

let modeloEntrenado;


const VALORES_ALTURAS = [1.82, 1.70, 1.87, 1.54, 1.63, 1.72];
const VALORES_PESOS = [80, 75, 85, 65, 72, 75];

const modelo = async () => {
  contenedorResultado.innerHTML = 'El modelo se esta entrenando...';

  // definir el tipo de modelo
  const model = tf.sequential();

  // definir las capas y cuantas neuronas va a tener
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // definir parametros
  // funcion de perdida: hace referencia a la forma de encontrar las equivocaciones cometidas en el procesamiento de los datos
  // optimizador: optimizar los valores de los parámetros para reducir el error cometido por la red
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // datos de entrada INPUT
  const xs = tf.tensor2d(VALORES_ALTURAS, [6, 1]);

  // datos de salidas OUTPUTS
  const ys = tf.tensor2d(VALORES_PESOS, [6, 1]);

  // entrenar modelo
  await model.fit(xs, ys, { epochs: 500 });

  btnCalcularPeso.disabled = false;
  btnCalcularPeso.focus();

  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
};

window.addEventListener('DOMContentLoaded', (event) => {
  modelo();

  btnCalcularPeso.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      const altura = parseFloat(btnCalcularPeso.value).toFixed(1);
      console.log(altura)
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([altura], [1, 1])
      );
      // console.log(resultado);
      const pesoResultado = resultado.dataSync();

      contenedorResultado.innerHTML = `El peso aproximado para la altura de: ${altura}: es de: ${pesoResultado}`;
    }
  });
});
