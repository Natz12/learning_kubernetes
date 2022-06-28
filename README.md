# learning_kubernetes

## Prerequisites

To use these instructions you will need to have docker and docker compose installed in your machine.

## Instructions

1. Run all the the scripts on install_kind folder. These will install kubectl, go, kind and helm.
2. Execute the following command to create the kubernetes cluster:

   `kind create cluster --name airflow-cluster --config /home/naty_kube/bash/kind_cluster.yaml`

- To debug yaml file install yamllint with the following command:

  `sudo apt-get update && sudo apt-get install yamllint -y`

  You can now use debug a yaml file with `yamlint <filename>`.

3. Create airflow namespace `kubectl create namespace airflow`
4. Fetch the official Helm chart of Apache Airflow that will get deployed on the cluster. We will add and update the official repository of the Apache Airflow Helm chart. Then deploy Airflow on Kubernetes with Helm install. The application will get the name airflow and the flag –debug allows us to check if anything goes wrong during the deployment:

```
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm search repo airflow
helm install airflow apache-airflow/airflow --namespace airflow --debug
```

We should get a similar output to this:

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

5. Check pods:

`kubectl get pods -n airflow`

6.  To access the Airflow UI, open a new terminal and execute the following command:

    `kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow --context kind-airflow-cluster`

    Recommendation: execute this command in a screen if you want to have permanent access to the Airflow webserver GUI through port 8080.

7.  We still need to modify the values.yaml file of helm. This file describes the configuration settings of our application such as the Airflow version to deploy, the executor to use, persistence volume mounts, secrets, environment variables and so on.

    To get this file we execute the command:

    `helm show values apache-airflow/airflow > values.yaml`

    We at least need to modify it to tell Airflow to use the KubernetesExecutor.

    `executor: "KubernetesExecutor"`

8.  Deploy Airflow in Kubernetes again:

    `helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug`

9.  to delete cluster: `kind delete cluster --name `

## Personalize Airflow

### Install dependencies with Airflow on Kubernetes

We have instances in which we need Airflow to interact with other dependencies and we might need to install some other Airflow providers like Spark, Docker, jenkins, mysql, etc. In this case we will need to build our own custom Docker image.

1. build the image by executing the command:

   `docker build -t airflow-custom:1.0.0`

2. Load the docker image into teh kubernetes cluster:

   `kind load docker-image airflow-custom:1.0.0 --name airflow-cluster`

3. Modify the file `values.yaml`:

```
defaultAirflowRepository: airflow-custom
defaultAirflowTag: "1.0.0"
```

4. Upgrade the Helm chart

```
helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug
helm ls -n airflow
```

5. Check available providers:

`kubectl exec <webserver_pod_id>`

# References:

1. [Airflow on Kubernetes : Get started in 10 mins](https://marclamberti.com/blog/airflow-on-kubernetes-get-started-in-10-mins/)
