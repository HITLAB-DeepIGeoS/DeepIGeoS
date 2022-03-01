if [ -d "$file" ]
then
    echo "$file found."
else
    export fileid=1Bv5Nc8lZOWdAeaX9uA5_uwUgGQ8sP96p
    export filename=BraTS2021_Training_Data.zip

    wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
         | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

    wget --load-cookies cookies.txt -O $filename \
         'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

    rm -f confirm.txt cookies.txt
    
    unzip $filename -d ./dataset | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'
fi
