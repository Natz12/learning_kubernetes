# learning_kubernetes

1. Run all the the scripts on install_kind folder. these will install kubectl, go, kind and helm.
2. run `kind create cluster --name airflow-cluster --config /home/naty_kube/bash/kind_cluster.yaml` to create the kubernetes cluster
- To debug taml file intall yamllint with `sudo apt-get update && sudo apt-get install yamllint -y` and then use with `yamlint <filename>'