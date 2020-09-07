# About

This repository represents my self-study learning about machine learning, and
deep learning in general. I will be working through the following materials:

* Basics of deep learning: http://ufldl.stanford.edu/tutorial/
* Manifold hypothesis: http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
* Unsupervised learning with generative adversarial networks: https://github.com/Newmu/dcgan_code
* Unsupervised learning with generative adversarial networks applied to NLP: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* A study of bias-varience tradeoff and cross-validation
* Read through a PhD thesis:  http://people.seas.harvard.edu/~dduvenaud/thesis.pdf

Although I have some experience with 'deep learning', with maximum likelihood
selection, I don't have much experience with neural networks, or formal
educaiton in deep learning. Just what I've encountered over the course of
earning my PhD in physics.

# Setting Up Local Development Environment

This setup is built on top of Docker to manage GPU dependencies. Docker has a
steep learning curve (for the uninitiated) as does setting up an environment
that manages the multiple driver dependencies that are a gigantic pain to
configure.

The basic idea of the setup is to accomplish the following steps:

* Install NVIDIA drivers on the machine with a CUDA compatible NVIDIA graphics
	card (Mine is running an NVIDIA GeForce GTX 1080 Ti, purchased on a whim
	during the height of the bitcoin rage, so you can sob along with me at the
	price...).
* Install docker, and set it up with the proper group permissions so you don't
	have to run it as root all the time - [official
	docs](https://docs.docker.com/engine/install/linux-postinstall/), for the
	lazy: `sudo groupadd docker; sudo usermod -aG docker $USER;` restart.

I've encuntered some issues with randomly losing my ability to run the
containers on system reboots, but I think this must have to do with 

1. Driver updates?
2. Load order (e.g. docker daemon starts before something that needs to start
	 first)

I'd have to troubleshoot more thoroughly to get a sterling set-up proceedure,
and there are probably gotchas along the way, but hopefully, this setup will
help get you started.
 
Docker removes the complexity of juggling drivers that can result in bricking
your system (not really bricked, but just a lot of pain of losing your GUI and
figuring out how to fix that) with the complexity of coming up with the arcane
docker-run commands that seem to be hundreds of lines long and completely
arbitrary (from the perspective of someone who hates reading man-pages).

## Install Proper Drivers

The first step of this procedure is to get the right NVIDIA drivers on your
system without breaking it. It used to be that you had to fuck around with
runfiles, custom drivers, and then turn off automatic driver updates because a
system update would break your computer.

The goal of this setup procedure is to reduce the complexity to

1. Install NVIDIA's proprietary drivers through the Ubuntu software center
2. Handle the rest with docker containers.

### Step 1: When you break your configuration by restarting your computer

After rebooting my system, attempting to restart the docker container created
errors becuase the GPU was no longer recognized by docker as a valid device.
This almost certainly means drives got updated and now everything is broken.

The docker stuff should all remain valid, assuming that we can get the GPU
drivers fixed. Following [this guide
here](https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/)
for a CLI driven setup. I did stuff up to the point of running the NVIDIA
script, restarted the docker daeomon and then stuff worked. Maybe all
subsequent steps are not necessary? /shrug.

## Step 2 - Install NVIDIA drivers with Ubuntu Software Center

If you don't know how to install proprietary drivers via ubuntu software
center, you're gonna have a bad time. Search Google and you will find answers.


After, confirm its working by running

```bash
nvidia-smi
```

To get an output such as:

```
Mon Aug 31 16:41:35 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   32C    P8    14W / 250W |    510MiB / 11176MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1359      G   /usr/lib/xorg/Xorg                            24MiB |
|    0      1496      G   /usr/bin/gnome-shell                          50MiB |
|    0      2406      G   /usr/lib/xorg/Xorg                           145MiB |
|    0      2544      G   /usr/bin/gnome-shell                         152MiB |
|    0      4018      G   ...AAAAAAAAAAAACAAAAAAAAAA= --shared-files   133MiB |
+-----------------------------------------------------------------------------+

```
## Set up Docker with GPU support

### Step 1

Search google for how to install Docker on an ubuntu machine (I'm using 18.04
LTS).

Set up docker, make sure your local user is added to the docker group,
[official docs](https://docs.docker.com/engine/install/linux-postinstall/).
This lets you run docker as your logged in user rather than as root.

### Step 2 - Get the NVIDIA docker container from Tensorflow

[Tensorflow](https://www.tensorflow.org/install/docker) has documentation for
accomplishing this, but the quick version is:

```bash
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

### Step 3 - Set up jupyter, link to local config

Jupyter needs to read config files locally, and docker by design separates your
local filesystem from the docker filesystem. We break this so that you can
reuse your local jupyter config:

This gets run once:
```bash
docker run -ti --gpus all --rm -u $(id -u):$(id -g) \
 -v "${HOME}/.config/jupyter:/.jupyter" \
 tensorflow/tensorflow:latest-gpu-jupyter \
 bash -c "source /etc/bash.bashrc && jupyter serverextension enable --py \
 jupyter_http_over_ws"
```

### Step 4 - Run the jupyter container

To access the jupyter container in your local browser, you need to tunnel the
port from the internal docker image out to your computer. Additionally, you
need to tell the docker image where your config for jupyter is, and where to
put notebooks on your local filesystem - otherwise everything lives *inside*
the docker container. In this case, we're using docker to handle drivers and
stuff, but we would prefer to use our local computer for version control, file
management, etc.

There's probably a fancy way to have docker load this information via some sort
of config file, but until I figure this out, its all gonna get passed in via a
fat command:

```bash
docker run -it -u $(id -u):$(id -g) \
  --gpus all -p 8888:8888 -v "${HOME}/.config/jupyter:/.jupyter" \
	-v "${HOME}/workspace:/tf" tensorflow/tensorflow:latest-gpu-jupyter \
	bash -c "source /etc/bash.bashrc && jupyter notebook \
	--notebook-dir=/tf --ip 0.0.0.0 --no-browser \
	--NotebookApp.allow_origin='https://colab.research.google.com'"
```

Obviously, this can only work if the folders you're giving the Docker image
access to actually exist. You have to have your directory structure set up -
for example, directorys like `${HOME}/workspace` and `${HOME}/.config/jupyter`
have to exist.

## Connecting colab.research.google.com to your local runtime

I like using google's colaboratory - I find its integration with Github, neat
plugins (like VIM keybindings, super-awesome form-syntax, hide/show code,
google drive integration) to be pretty nifty. There's no reason to use Colab if
you don't like it - but otherwise, you have to get your jupyter notebook
configuration, plugins, etc all set up in your docker image. This is a
short-cut to reasonable defaults with the added benefit of being able to play
around with your notebooks anywhere, even if you don't have access to your
development machine.

You can put your development machine behind a dyndns server to roll your own
cloud environment and connect it to some domain service, but that's a tutorial
for another day.

### Step 1: Connect local runtime to colab

Go to colab, and connect to a local runtime, pasting the string that looks like
this:

`http://127.0.0.1:8888/?token=<long alphanumeric token>` into place for the
backend URL, but replace `127.0.0.1` with `localhost`, since you tunneled the
port earlier when you wrote `-p 8888:8888` in your docker run command.

### Step 2: Create some scripts to run these horrible commands for you

In this repo, you can see a `scripts` directory which runs all the beastly
docker scripts for you. If you have a bad memory like me, it might make sense
to copy these long shitty commands into a script so that you can run the
container environment by running the scripts instead of remembering everything
you have to type in. 

### Step 3: In your connected colab session, test your environment

[Run the notebook to test your environment]('notebooks/Testing Your
Environment.ipynb').

For the lazy:

```python
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

Returns:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Modify your Docker image to suit your needs

Now that you have a Docker image which supports GPU-accellerated tensorflow
(thanks, Google, Docker, and friendly internet strangers that are far more
patient than I am!), you might want to modify it to suit your needs.

While colab has this awesome ability to install libraries in real time, its
kind of a pain to reinstall sklearn or matplotlib every time you restart your
docker image. In fact, Docker has this amazing built-in ability to take an
existing docker image and add your own dependencies to it. Incredible!

Anyway, here's what you do:

### Step 1: Setup Additional Dependencies

You might want to add stuff like pandas, matplotlib, seaborn, etc, to create a
more full-featured analysis environment. You can do so by building a dockerfile
based on another docker container.

Collect all your additional dependencies in a	`requirements.txt` file that
exists in the same directory as a `Dockerfile`. We will write both the
`Dockerfile` and the `requirements.txt` files ourselves.

```
# Contents of requirements.txt
pandas
sklearn
matplotlib
seaborn
```

We'll create a Dockerfile that piggy-backs on top of the tensorflow jupyter
image:

```
# Contents of Dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY requirements.txt /tmp
WORKDIR /tmp
RUN python -m pip install -r requirements.txt
```

### Step 2: Build your custom Docker runtime
Put whatever you want to install in tensorflow image with pip in the
requirements.txt file.

In the directory that contains your `Dockerfile` and `requirements.txt`, build
the new docker image, giving it a target name (`-t tf_gpu_extra`) that makes
sense to you.

```bash
$ docker build -t tf_gpu_extra .
```

[Celebrate!](https://www.youtube.com/watch?v=3GwjfUFyY6M)

## Useful Links:

During my research on how to set this up properly, I found the following links
to be useful

* https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html
	* Running jupyter notebooks using the docker CLI
* https://docs.docker.com/engine/reference/run/
	* General docs on the Docker engine
* https://stackoverflow.com/questions/61024722/how-to-use-google-colab-with-a-local-tensorflow-jupyter-server-using-powershell
	* Some info about how to run docker, with tensorflow, with local colab runtimes.
* https://github.com/tensorflow/tensorflow/issues/25247#issuecomment-459644861
	* Magic docker commands to point a runtime to your jupyter config, local
		filesystem, and port-forward.
* https://stackoverflow.com/questions/58191215/how-to-add-python-libraries-to-docker-image
	* Modifying a docker image to add additional python dependencies
* https://research.google.com/colaboratory/local-runtimes.html
	* Google's documentation for how to set up a local colab runtme.

