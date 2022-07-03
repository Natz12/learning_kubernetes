# learning_kubernetes

## Prerequisites

To use these instructions you will need to have docker and docker compose installed in your machine.

## Instructions

1.  Run all the the scripts on install_kind folder. These will install kubectl, go, kind and helm.

2.  Execute the following command to create the kubernetes cluster:

    `kind create cluster --name airflow-cluster --config <path>/cluster_config/deploymnets/kind_cluster.yaml`
    <!-- kind create cluster --image kindest/node:v1.18.15 -->

    - To debug yaml file install yamllint with the following command:

    `sudo apt-get update && sudo apt-get install yamllint -y`

    You can now use debug a yaml file with `yamlint <filename>`.

3.  Create airflow namespace

    ```
    export NAMESPACE=airflow-namespace
    export RELEASE_NAME=airflow-releasename
    kubectl create namespace $NAMESPACE
    ```

4.  Change the default namespace (optional)

    `kubectl config set-context --current --namespace=$NAMESPACE`

5.  Fetch the official Helm chart of Apache Airflow that will get deployed on the cluster. We will add and update the official repository of the Apache Airflow Helm chart. Then deploy Airflow on Kubernetes with Helm install. The application will get the name airflow and the flag –debug allows us to check if anything goes wrong during the deployment:

    ```
    helm repo add apache-airflow https://airflow.apache.org
    helm repo update
    helm search repo airflow
    helm install $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE --debug
    ```

    to install the chart with example dags:

    ```
    helm install $RELEASE_NAME apache-airflow/airflow \
     --namespace $NAMESPACE \
     --set-string "env[0].name=AIRFLOW__CORE__LOAD_EXAMPLES" \
     --set-string "env[0].value=True"
    ```

    We should get a similar output to this:

    ```
    Your release is named airflow.
    You can now access your dashboard(s) by executing the following command(s) and visiting the corresponding port at localhost in your browser:

    Airflow Webserver: kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
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

    # WARNING: You should set a static webserver secret key

    ###########################################################

    You are using a dynamically generated webserver secret key, which can lead to
    unnecessary restarts of your Airflow components.

    Information on how to set a static webserver secret key can be found here:
    https://airflow.apache.org/docs/helm-chart/stable/production-guide.html#webserver-secret-key

    ```

6.  Check pods are up:

    `kubectl get pods --namespace $NAMESPACE`

7.  To access the Airflow UI, open a new terminal and execute the following command:

    `kubectl port-forward svc/airflow-webserver 8080:8080 -n $NAMESPACE --context kind-airflow-cluster`

    `kubectl port-forward svc/airflow-releasename-webserver 8080:8080 -n $NAMESPACE --context kind-airflow-cluster`

    Recommendation: execute this command in a screen if you want to have permanent access to the Airflow webserver GUI through port 8080.

8.  We still need to modify the values.yaml file of helm. This file describes the configuration settings of our application such as the Airflow version to deploy, the executor to use, persistence volume mounts, secrets, environment variables and so on.

To get this file we execute the command:

`helm show values apache-airflow/airflow > values.yaml`

<!--
We at least need to modify it to tell Airflow to use the KubernetesExecutor.

`executor: "KubernetesExecutor"`

9. Deploy Airflow in Kubernetes again:

`helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug` -->

10. to delete cluster: `kind delete cluster --name airflow-cluster`

## Personalize Airflow

### Install dependencies with Airflow on Kubernetes

We have instances in which we need Airflow to interact with other dependencies and we might need to install some other Airflow providers like Spark, Docker, jenkins, mysql, etc. In this case we will need to build our own custom Docker image.

1. build the image by executing the command:
<!-- 
`docker build . -t airflow-custom:1.0.0`

`docker build --pull -t airflow-custom:1.0.3 .` -->

`docker build --pull --tag my-dags:0.0.1 .`

2. Load the docker image into teh kubernetes cluster:

<!-- `kind load docker-image airflow-custom:1.0.0 --name airflow-cluster`
`kind load docker-image airflow-custom:1.0.3` -->

`kind load docker-image my-dags:0.0.1 --name airflow-cluster`

3. Upgrade Helm deployment

<!-- ```

helm upgrade $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE \
 --set images.airflow.repository=airflow-custom \
 --set images.airflow.tag=1.0.3 --debug

``` -->

```
helm upgrade $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE \
 --set images.airflow.repository=my-dags \
 --set images.airflow.tag=0.0.1 --debug
```

<!-- 3. Modify the file `values.yaml`:

```

defaultAirflowRepository: airflow-custom
defaultAirflowTag: "1.0.0"

```

???? https://youtu.be/AjBADrVQJv0?t=1025

```

env:

- name: "AIRFLOW**KUBERNETES**WORKER_CONTAINER_REPOSITORY"
  value: "hail_monitoring_goes"
- name: "AIRFLOW**KUBERNETES**WORKER_CONTAINER_TAG"
  value: "latest"

```

????

4. Upgrade the Helm chart

```

helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug
helm ls -n airflow

```-->

5. Check available providers:

`kubectl exec <webserver_pod_id> -- airflow providers list`

6. To see docker-images that are loaded

   6.1 Get name of a node by running `kubectl get nodes`.

   6.2 Get into the node by running `docker exec -ti <nodename> bash`

   6.3 After getting into the node you can just run `crictl images` to see images loaded on that node.

## Mounting DAGS from private github

https://airflow.apache.org/docs/helm-chart/stable/manage-dags-files.html#mounting-dags-from-a-private-github-repo-using-git-sync-sidecar

ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

```
helm upgrade --install airflow apache-airflow/airflow -f /home/naty_kube/git/learning_kubernetes/cluster_config/deployments/override-values.yml  --set images.airflow.repository=my-dags --set images.airflow.tag=0.0.1 --debug
```

```
helm install $RELEASE_NAME apache-airflow/airflow \
    --namespace $NAMESPACE -f /home/naty_kube/git/learning_kubernetes/cluster_config/deployments/override-values.yml \
    --set-string "env[0].name=AIRFLOW__CORE__LOAD_EXAMPLES" \
    --set-string "env[0].value=True"  --debug
```

`kubectl create secret generic airflow-ssh-git-secret --from-file=gitSshKey=</Your/path/.ssh/id_rsa>`

`kubectl create secret generic airflow-ssh-git-secret --from-file=gitSshKey=/home/naty_kube/.ssh/id_rsa`

`kind load docker-image my-dags:0.0.1 --name airflow-cluster`

```
helm upgrade $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE \
 --set images.airflow.repository=my-dags \
 --set images.airflow.tag=0.0.1 --debug
```

## Add DAGS to Docker image

1. Add the following line to the Dockerfile

`COPY dags $AIRFLOW_HOME/dags`

AIRFLOW_HOME environment variable `airflowhome` can be found in the values.yaml file

## Deploy DAGs on Kubernetes with GitSync

1. we are going to need an SSH key to login into github. See the documentation to [Generate an SSH key to connect to github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

2. modify the values.yaml as follows:

```

gitSync:
enabled: true
repo: ssh://git@github.com/<github_use>/airflow-2-dags.git
branch: main
rev: HEAD
root: "/git"
dest: "repo"
depth: 1
subPath: ""
sshKeySecret: airflow-ssh-git-secret

```

3. Create a secret with our private key on it. With Kubectl, the value is automatically encoded in Base64

`kubectl create secret generic airflow-ssh-git-secret --from-file=gitSshKey=</Your/path/.ssh/id_rsa>`

To check the secret was created:
`kubectl get secrets`

4. Upgrade Airflow in Kubernetes again:

`helm upgrade --install airflow apache-airflow/airflow -n airflow -f values.yaml --debug`

## Set up persistent Airflow logs

1. defaultAirflowRepository

## Helpful for debug

`kubectl describe pods > temp.txt`

`kubectl run <pod_name> --image=<image>:<tag>`

## References:

1. [Airflow on Kubernetes : Get started in 10 mins](https://marclamberti.com/blog/airflow-on-kubernetes-get-started-in-10-mins/)

```

```
