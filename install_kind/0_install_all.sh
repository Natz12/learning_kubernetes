#!/usr/bin/bash

# echo $PWD

# echo ${BASH_SOURCE[0]}
DIR=${BASH_SOURCE%/*}
for VARIABLE in 1_kubectl.sh 2_go.sh 3_kind.sh 4_helm.sh
do
    echo "Installing " $VARIABLE
    chmod +x $DIR/$VARIABLE
    $DIR/$VARIABLE
done
