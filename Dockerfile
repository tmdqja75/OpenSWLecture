# set base image
FROM pytorch/pytorch

RUN python -m pip install --upgrade pip

RUN pip install \
        sklearn \
        flask \
        pandas 

# Set starting working path directory (when container is started)
WORKDIR /workspace

ADD app_torch.py /workspace
ADD templates /workspace/templates
ADD tmp_pytorch /workspace/tmp_pytorch
ADD tmp /workspace/tmp

