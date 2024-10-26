#!/bin/bash

EXPORT_COMMAND='export LD_LIBRARY_PATH=`python3 -c "import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))"`'

if ! grep -Fxq "$EXPORT_COMMAND" /root/.bashrc; then
    echo "$EXPORT_COMMAND" >> /root/.bashrc
    echo "Added LD_LIBRARY_PATH update to /root/.bashrc."
else
    echo "LD_LIBRARY_PATH update is already present in /root/.bashrc."
fi
