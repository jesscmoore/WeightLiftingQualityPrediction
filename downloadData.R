# downloadData.R


## Data download and unzip 
# string variables for file download
filetrain <- "pml-training.csv"
filetest <- "pml-testing.csv"
urltrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Training dataset
if(!file.exists(filetrain)){
  cat("Downloading training data file...\n")
  download.file(urltrain,filetrain) # Use "wb" mode for binary file downloads
}

# Test dataset
if(!file.exists(filetest)){
  cat("\nDownloading testing data file...\n")
  download.file(urltest,filetest) # Use "wb" mode for binary file downloads
}


