# DAQN (Deep Autoencoder Q-network) 1.0

Este proyecto contiene el código base de una DAQN, es decir, una DQN que posee un autoencoder como parte de pre-procesado.
El código además incluye un algoritmo DQN básico (aumentado con memoria de reproducción) y un algoritmo DQN con LSTM (en desarrollo).

## Instrucciones de instalación

1. Primero crear un entorno virtual en la carpeta AI con ```virtualenv ```.

2. Entrar al entorno virtual con ```venv/scripts/``` en windows o ```. venv/bin/activate```

3. Elegir el setup.py acorde al sistema operativo.
	a. Si usa windows borre "setup_linux.py" y cambie el nombre de "setup_win.py" a "setup.py"
	a. Si usa windows borre "setup_win.py" y cambie el nombre de "setup_linux.py" a "setup.py"

3. Ejecutar ```pip install --editable ```

4. Si usa windows debe instalar pytorch, torchvision y vizdoom con pip.

## Uso

1. Activar visdom escribiendo justamente ```visdom``` en el terminal, sino no se
verán los gráficos. (para verlo se debe ir al navegador y escribir localhost:8097).

2. Ejectutar AI.py simplemente escribiendo ```AI train``` en la consola o terminal. Con esto se ejecutará
el programa de entrenamiento en el mapa más básico (basic.cfg) con el algoritmo DQN básico, que solamente posee 
memoria de reproducción.

## Entrenamiento de autoencoder

Usualmente para usar la DAQN basta con utilizar AI train --daqn, sin embargo si se desea entrenar al
autoencoder desde cero se puede hacer lo siguiente.

1. Para obtener un dataset hay dos formas, una es utilizando el comando ```AI imcapture```, el cual captura un
dataset desde un agente que realiza acciones aleatorias.
Esta función tiene los siguientes parámetros:

a. --config_path. Valor por defecto = 'basic.cfg'. Este parámetro se usa para definir el archivo de configuración
a utilizar, lo que a su vez define la configuración del juego y el mapa. Ejemplo: &nbsp;
```AI imcapture --config_path=defend_the_center.cfg```

b. --episodes. Valor por defecto = 1000. Define la cantidad de episodios a correr. &nbsp;
```AI recording --episodes=10000```

c. --image_limit. Valor por defecto = 0 (no hay limite). Define la cantidad de imágenes a tomar, una
vez se llega a ese límite el programa finaliza. &nbsp;
```AI recording --image_limit=10000```

La segunda forma para obtener un dataset es grabando un replay y después reproduciendolo para obtener un dataset de experiencia humana.
Para ello se puede utilizar el comando  ```AI recording``` para grabar y despues utilizar  ```AI watchreplay```.
Al hacer esto se guardará cada frame del replay como parte del dataset. &nbsp;

```AI recording``` tiene los siguientes parámetros:

a. --replay_file. Valor por defecto = replay.lmp. Define el nombre del archivo de replay para guardar. Ejemplo:  &nbsp;
```AI recording --replay_file=guadado.lmp```

b. --config_file. Valor por defecto = 'basic.cfg'. Este parámetro se usa para definir el archivo de configuración
a utilizar, lo que a su vez define la configuración del juego y el mapa. Ejemplo:  &nbsp;
```AI recording --config_path=defend_the_center.cfg```

```AI watchreplay``` tiene los siguientes parámetros:

a. --replay_file. Valor por defecto = replay.lmp. Define el nombre del archivo de replay para cargar y ver. Ejemplo:  &nbsp;
```AI watchreplay --replay_file=guadado.lmp```

2. Ejecutar el comando ```AI ae_train```

3. Probar el algoritmo DAQN con comando ```AI train --daqn```

## Parámetros AI train

1. --learning_rate. Valor por defecto = 0.00025. Esto sirve para definir la tasa de aprendizaje de la función Q. Ejemplo: 
```AI train --learning_rate=0.02```

2. --discount_factor. Valor por defecto = 0.99. Este parámetro sirve para definir el factor de descuento en la función Q. Ejemplo: 
```AI train --discount_factor=0.8```

3. --config_path. Valor por defecto = 'basic.cfg'. Este parámetro se usa para definir el archivo de configuración
a utilizar, lo que a su vez define la configuración del juego y el mapa. Ejemplo: 
```AI train --config_path=deathmatch.cfg```

4. --model_to_load. No posee valor por defecto, ya que si está vacio se empieza un entrenamiento desde cero. Define el modelo
de red neuronal a cargar que haya sido previamente entrenada. Estos modelos se van guardando a cada época en la carpeta "AI/nets". Ejemplo: 
```AI train --model_to_load=./nets/model_19.pth```

5. --skip_learning. Es un flag que define si se salta la fase de aprendizaje para pasar directamente a la fase de pruebas y ver
al agente jugar. Ejemplo: 
```AI train --skip_learning```

6. --skip_watching. Este flag define si saltarse la parte de pruebas y solo realizar el entrenamiento. Especialmente útil si se
dejará un comando por lotes entrenando al agente varias veces. Ejemplo: 
```AI train --skip_watching```

7. --image_capture. Este flag define si se harán capturas de pantalla para agregar al dataset del autoencoder. Las imágenes se van
guardando en la carpeta "AI/training_set". Ejemplo: 
```AI train --image_capture```

8. --daqn. Este flag define si se utilizara el modelo DAQN, que es el que posee autoencoders. Ejemplo: 
```AI train --daqn```

9. --drqn. Este flag define si se utilizará el modelo DQN con LSTM. Ejemplo: 
```AI train --drqn```

## Parámetros AI ae_train

1. --learning_rate. Valor por defecto = 1e-3. Esto sirve para definir la tasa de aprendizaje del autoencoder. Ejemplo: 
```AI ae_train --learning_rate=0.02```

2. --load_model. Este flag define sirve para definir que se utilizará el autoencoder ya entrenado en vez de crear uno de cero. Ejemplo: 
```AI ae_train --load_model```

3. --epochs. Este valor es un entero que define cuantas épocas de entrenamiento habrá. Ejemplo:
```AI ae_train --epochs=100```

## Notas

- El archivo sgtlink.py funciona, sin embargo el token de slack fue desactivado
por lo que es recomendable comentar el código de slack o obtener un token propio para
copiarlo en el código.

- Se puede cargar un agente ya entrenado utilizando el comando --model_to_load='./nets/model_19.pth',
que es un agente entrenado por 20 épocas y para verlo en acción inmediatamente hay que agregar --skip_learning.
