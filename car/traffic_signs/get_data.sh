wget https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip 
unzip *.zip
rm *.zip
# the zip has data in a lab* folder, move it to data
mv lab* data
cd data
rm *docx

# Enter data and rename *.p files in *.pkl files
for file in *.p
do 
	mv "$file" "${file%.p}.pkl"
done

