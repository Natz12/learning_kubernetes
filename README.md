# learning_kubernetes

1. Run all the the scripts on install_kind folder. These will install kubectl, go, kind and helm.
2. run `kind create cluster --name airflow-cluster --config /home/naty_kube/bash/kind_cluster.yaml` to create the kubernetes cluster
- To debug yaml file install yamllint with `sudo apt-get update && sudo apt-get install yamllint -y` and then use with `yamlint <filename>'