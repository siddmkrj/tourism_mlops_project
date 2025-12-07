# Tourism MLOps Project Setup

This document records the steps I followed to set up the Tourism MLOps project.

## Prerequisites

- Conda installed on my system
- A Hugging Face account

## Setup Steps

### 1. Created Conda Environment

I created and activated a conda environment:

```bash
conda create -n tourism-mlops python=3.10 -y
conda activate tourism-mlops
```

### 2. Installed Requirements

I installed the project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up Hugging Face

#### Installed Hugging Face CLI

I installed the Hugging Face CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

#### Created Hugging Face Account & Token

I completed the following steps:

1. Went to [huggingface.co](https://huggingface.co) and signed in / signed up
2. Clicked my profile → Settings → Access Tokens
3. Created a New token (type: Write) and copied it

#### Logged In from Terminal

I logged in from the terminal (inside the conda environment):

```bash
huggingface-cli login
```

I pasted my token when prompted.

#### Created Dataset Repository on Hugging Face

I created the dataset repository in my browser:

1. Went to [huggingface.co/datasets](https://huggingface.co/datasets)
2. Clicked **New dataset**
3. Named it: `mukherjee78/tourism-wellness-package`
4. Set visibility to **Public**
5. Clicked **Create**


### 4. Created Project Structure

I created the project folder structure:

```bash
mkdir data notebooks src
```

Then I:
1. Created a notebook inside the `notebooks` folder
2. Copied the `tourism.csv` file into the `data` folder