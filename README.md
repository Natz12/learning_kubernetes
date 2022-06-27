# learning_kubernetes

## Prerequisites

To use these instructions you will need to have docker and docker compose installed in your machine.

## Instructions

1. Run all the the scripts on install_kind folder. These will install kubectl, go, kind and helm.
2. Execute the following command to create the kubernetes cluster:
   `kind create cluster --name airflow-cluster --config /home/naty_kube/bash/kind_cluster.yaml`

- To debug yaml file install yamllint with the following command:
  `sudo apt-get update && sudo apt-get install yamllint -y`
  Yo ucan now use it with `yamlint &lt;filename&gt;'.

3. Create airflow namespace `kubectl create namespace airflow`
4. Fetch the official Helm chart of Apache Airflow that will get deployed on the cluster. We will add and update the official repository of the Apache Airflow Helm chart. Then deploy Airflow on Kubernetes with Helm install. The application will get the name airflow and the flag –debug allows us to check if anything goes wrong during the deployment:

```
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm search repo airflow
helm install airflow apache-airflow/airflow --namespace airflow --debug
```

We get a similar output to this:

```
Your release is named airflow.
You can now access your dashboard(s) by executing the following command(s) and visiting the corresponding port at localhost in your browser:

Airflow Webserver:     kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
Default Webserver (Airflow UI) Login credentials:
    username: admin
    password: admin
Default Postgres connection credentials:
    username: postgres
    password: postgres
    port: 5432

You can get Fernet Key value by running the following:

    echo Fernet Key: $(kubectl get secret --namespace airflow airflow-fernet-key -o jsonpath="{.data.fernet-key}" | base64 --decode)

###########################################################
#  WARNING: You should set a static webserver secret key  #
###########################################################

You are using a dynamically generated webserver secret key, which can lead to
unnecessary restarts of your Airflow components.

Information on how to set a static webserver secret key can be found here:
https://airflow.apache.org/docs/helm-chart/stable/production-guide.html#webserver-secret-key
```

5. To access the Airflow UI, open a new terminal and execute the following command:
   `kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow --context kind-airflow-cluster`
   Recommendation: execute this command in a screen if you want to have permanent access to the Airflow webserver GUIthrough port 8080.

6. We still need to modify the values.yaml file of helm. This file describes the configuration settings of our application such as the Airflow version to deploy, the executor to use, persistence volume mounts, secrets, environment variables and so on.
   To get this file we execute the command:
   `helm show values apache-airflow/airflow > values.yaml`

   We at least need to modify it to tell Airflow to use the KubernetesExecutor.
   `executor: "KubernetesExecutor"`

7. Deploy Airflow in Kubernetes again:
   `helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug`

8. to delete cluster: `kind delete cluster --name `yamlint &lt;cluster name&gt;'
