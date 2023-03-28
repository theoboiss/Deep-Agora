mkdir raw_page_dataset
cd raw_page_dataset


wget -N -q --show-progress -P ./ https://zenodo.org/record/257972/files/Train%20-%20Baseline%20Competition%20-%20Simple%20Documents.tar.gz

tar --keep-newer-files -xzf *.tar.gz -C ./ 2>/dev/null

cp "Baseline Competition - Simple Documents"/ABP_FirstTestCollection/M_Aigen_am_Inn_002-01_0000.jpg ./
cp "Baseline Competition - Simple Documents"/ABP_FirstTestCollection/M_Aigen_am_Inn_007_0084.jpg ./
cp "Baseline Competition - Simple Documents"/ABP_FirstTestCollection/page/M_Aigen_am_Inn_002-01_0000.xml ./
cp "Baseline Competition - Simple Documents"/ABP_FirstTestCollection/page/M_Aigen_am_Inn_007_0084.xml ./

rm -r "Baseline Competition - Simple Documents"
rm "Train - Baseline Competition - Simple Documents.tar.gz"


cd ../