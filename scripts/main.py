# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""
import os
from src.config import PROC_MMS_DIR
from src.processing.reading import import_processed_data

mms_dir = os.path.join(PROC_MMS_DIR, '1min')
mms = import_processed_data(mms_dir)