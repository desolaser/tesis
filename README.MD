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

## Parámetros

1. --learning_rate. Valor por defecto = 0.00025. Esto sirve para definir la tasa de aprendizaje de la función Q. Ejemplo: 
```AI train --learning_rate=0.02```

2. --discount_factor. Valor por defecto = 0.99. Este parámetro sirve para definir el factor de descuento en la función Q. Ejemplo: 
```AI train --discount_factor=0.8```

3. --config_path. Valor por defecto = '../scenarios/basic.cfg'. Este parámetro se usa para definir el archivo de configuración
a utilizar, lo que a su vez define la configuración del juego y el mapa. Ejemplo: 
```AI train --config_path=../scenarios/deathmatch.cfg```

4. --model_to_load. No posee valor por defecto, ya que si está vacio se empieza un entrenamiento desde cero. Define el modelo
de red neuronal a cargar que haya sido previamente entrenada. Estos modelos se van guardando a cada época en la carpeta "AI/nets". Ejemplo: 
```AI train --model_to_load=./nets/model_19.pth```

5. --skip_learning. Es un flag que define si se salta la fase de aprendizaje para pasar directamente a la fase de pruebas y ver
al agente jugar. Ejemplo: 
```AI train --skip_learning.```

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

## Entrenamiento de autoencoder

Usualmente para usar la daqn basta con utilizar AI train --daqn, sin embargo si se desea entrenar al
autoencoder desde cero se puede hacer lo siguiente.

1. Para obtener un dataset es posible utilizar el comando "AI train --image_capture" en varios mapas. Ejemplos:
```AI train --config_path=../scenarios/defend_the_center.cfg --image_capture```
```AI train --config_path=../scenarios/deathmatch.cfg --image_capture```
```AI train --config_path=../scenarios/rocket_basic.cfg --image_capture```

2. Crear otra carpeta en AI llamada "training_set_min/train/" y copiar el contenido de
"training_set" a la carpeta creada anteriormente.

3. Utilizar resizer.py para cambiar el tamaño de las imágenes a 108 x 60. (Dentro
del código se puede definir el tamaño final)

4. Ejecutar el archivo "python3 ae_trainer.py".

5. Al terminal saldrá un archivo llamado "autoencoder.pth". Este archivo debe ser
copiado a la carpeta "AI/src/model/".

6. Ejecutar ```AI train --daqn```.

## Notas

- El archivo sgtlink.py funciona, sin embargo el token de slack fue desactivado
por lo que es probable que tenga algunos problemas.

- Se puede cargar un agente ya entrenado utilizando el comando --model_to_load='./nets/model_19.pth',
que es un agente entrenado por 20 épocas y para verlo en acción inmediatamente hay que agregar --skip_learning.
