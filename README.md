This algorithm was designed to perform lung segmentation on 2D CT scans from the "Chest XRay Masks and Labels" Dataset on Kaggle (https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels?select=Lung+Segmentation) using the UNet architecture (https://arxiv.org/pdf/1505.04597). 


### Create and Activate a Virtual Environment
Create a virtual environment (e.g., `.venv`) and activate it.

For example, use: 
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
Install the project dependencies using:
```bash
pip install -e .
```
Then do: 
```bash
pip install -r requirements.txt
```
