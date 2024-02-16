# Deep learning applications
Welcome to this lab in the computer vision course [TEK5030] at the University of Oslo.
In this lab we will play with pre-trained models that are available in the [Model Zoo For OpenCV DNN][model zoo].

Make sure to check the [prerequisites](#install-git-lfs) before getting started. 

[TEK5030]: https://www.uio.no/studier/emner/matnat/its/TEK5030/
[the intro lab]: https://github.com/tek5030/lab-intro/tree/master/py
[model zoo]: https://github.com/opencv/opencv_zoo

---

**Start** by cloning this repository on your machine.

Initialize the Python environment

```bash
# Clone the lab
git clone https://github.com/tek5030/lab-dnn.git

cd lab-dnn

python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m ipykernel install --user --name=venv
```

**Then**, download the `opencv_zoo` within the project directory.

```bash
# Pull the model zoo
git clone --depth 1 https://github.com/opencv/opencv_zoo && cd opencv_zoo
git lfs install
git lfs pull
```

**Open** the project in PyCharm.
If you are uncertain about how this is done, please take a look at [the intro lab].

The lab is carried out by following these steps:

1. [Get an overview][first step]
2. [Play around with examples from the OpenCV model zoo][second step]
3. [Play around with examples from the OpenCV dnn tutorials][third step]
4. [Further work][last step]

Please start the lab by going to the [first step].

[first step]: lab-guide/1-get-an-overview.md
[second step]: lab-guide/2-model-zoo.md
[third step]: lab-guide/3-opencv-tutorials.ipynb
[last step]: lab-guide/4-further-work.md

---

## Prerequisites
If you are on a lab computer, you are all set.

If you are on Ubuntu and haven't completed [the intro lab], the following should be sufficient _for this lab_.

### Install required Python packages
```bash
sudo apt update
sudo apt install python3 python3-dev python3-distutils python3-venv python-is-python3
```

### Install git lfs

The OpenCV model zoo repository requires [git lfs](https://git-lfs.com/) (Git Large File System):

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs
```
