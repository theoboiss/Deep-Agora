mkdir raw_datasets
cd raw_datasets

wget -N -q --show-progress -P ./ https://zenodo.org/record/257972/files/Train%20-%20Baseline%20Competition%20-%20Simple%20Documents.tar.gz

wget -N -q --show-progress -P ./ https://zenodo.org/record/257972/files/Train%20-%20Baseline%20Competition%20-%20Complex%20Documents.tar.gz

wget -N -q --show-progress -P ./ https://zenodo.org/record/3945088/files/FCR_500.tar.gz

tar --keep-newer-files -xzf *.tar.gz -C ./ 2>/dev/null


cd ../