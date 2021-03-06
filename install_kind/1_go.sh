#!/bin/bash
tput setaf 5 
echo -e "\n*******************************************************************************************************************"
echo -e "Installing Go"
echo -e "*******************************************************************************************************************"
tput setaf 2

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Downloading Go archive"
echo -e "*******************************************************************************************************************"
tput setaf 2
wget https://dl.google.com/go/go1.14.2.linux-amd64.tar.gz

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Extracting archive"
echo -e "*******************************************************************************************************************"
tput setaf 2
tar -xzf go1.14.2.linux-amd64.tar.gz

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Cleaning up tar file"
echo -e "*******************************************************************************************************************"
tput setaf 2
rm go1.14.2.linux-amd64.tar.gz

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Moving Go binary to /usr/local"
echo -e "*******************************************************************************************************************"
tput setaf 2
sudo mv go /usr/local
echo -e "Go binary move complete"

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Adding Go environment variables to bash_profile"
echo -e "*******************************************************************************************************************"
tput setaf 2
cat << 'EOF' >> ~/.profile
export GOROOT=/usr/local/go
export GOPATH=~/go/kind
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
EOF
echo -e "Variables added"

tput setaf 5
echo -e "\n*******************************************************************************************************************"
echo -e "Refreshing profile to load variables"
echo -e "*******************************************************************************************************************"
tput setaf 2
source ~/.profile
echo -e "Profile has been refreshed"

tput setaf 3
echo -e "\n*******************************************************************************************************************"
echo -e "Go installation complete"
echo -e "*******************************************************************************************************************"
tput setaf 2
