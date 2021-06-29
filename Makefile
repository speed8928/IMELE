# $ make build 
# $ make run

export CSV_DATA=dataset/data.csv

build: 
	docker build -f Dockerfile -t imele .

run:
	docker run -it --rm --gpus all \
      --shm-size=18300m \
      -v `pwd`:/work/ imele \
	  python test.py --model ./ --csv $(CSV_DATA) --outfile ./

# launch jupyter notebook
jn:
	docker run -it --rm --gpus all \
      --shm-size=18000m \
      -v `pwd`:/work/ \
	  -p 8888:8888 \
	  imele \
      jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root

# launch docker with bash
void: 
	docker run -it --rm --gpus all \
      -v `pwd`:/work/ imele \
      /bin/bash

# for installing pytorch-ssim
install-pytorch-ssim:
	git clone https://github.com/Po-Hsun-Su/pytorch-ssim.git