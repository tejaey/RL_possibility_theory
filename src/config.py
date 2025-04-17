from dotenv import load_dotenv
import torch
import os

load_dotenv()

DEVICE = torch.device(os.getenv("DEVICE", "cpu"))
