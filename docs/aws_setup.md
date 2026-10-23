# AWS setup for continuous delivery (one-time)

Account `793239960389`, region `ca-central-1` (Montréal). Target: **Amazon ECS Express Mode on Fargate**.

## Why not App Runner
The course specification names AWS App Runner, but AWS no longer offers App Runner to new customers
(https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html), and the UI
framework (Streamlit) depends on WebSockets, which App Runner is reported not to support. AWS recommends
ECS Express Mode as the replacement: the same "give it an image, get an HTTPS URL" workflow on
Fargate behind an Application Load Balancer. This deviation is documented in the D4 deployment note.

## 0. Cost guard (do this first)
Billing and Cost Management → Budgets → Create budget → "Zero spend" or a monthly cost budget of **$10**,
with an email alert at 80% and 100%. The running system is not free (Fargate tasks plus a load balancer);
check current prices on the AWS pricing pages and delete the services when the course is over (section 4).

## 1. Create the AWS resources (AWS CloudShell)
Open the AWS console in `ca-central-1`, click the CloudShell icon (top bar), and paste this block.
Lines that may already exist end with `|| true`, so re-running is safe.

```bash
export AWS_DEFAULT_REGION=ca-central-1
ACCOUNT=793239960389
REPO=AI-DS-Git/boreal-marche-customer-intelligence

# Image repositories (scan on push)
aws ecr create-repository --repository-name boreal-api --image-scanning-configuration scanOnPush=true || true
aws ecr create-repository --repository-name boreal-ui  --image-scanning-configuration scanOnPush=true || true

# Service-linked roles and the cluster (as admin, so the deploy role does not need these rights)
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com || true
aws iam create-service-linked-role --aws-service-name elasticloadbalancing.amazonaws.com || true
aws ecs create-cluster --cluster-name default || true

# GitHub OIDC identity provider
aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 || true

# Task execution role (pulls images, writes logs)
cat > trust-tasks.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document file://trust-tasks.json || true
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Infrastructure role (lets Express Mode create the load balancer and networking pieces)
cat > trust-infra.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name ecsInfrastructureRoleForExpressServices --assume-role-policy-document file://trust-infra.json || true
INFRA_POLICY=$(aws iam list-policies --scope AWS \
  --query "Policies[?PolicyName=='AmazonECSInfrastructureRoleforExpressGatewayServices'].Arn" --output text)
echo "infrastructure policy: $INFRA_POLICY"
aws iam attach-role-policy --role-name ecsInfrastructureRoleForExpressServices --policy-arn "$INFRA_POLICY"

# Role GitHub Actions assumes. Only workflows running on the main branch of this repository can use it.
cat > trust-github.json <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Federated":"arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"},
 "Action":"sts:AssumeRoleWithWebIdentity",
 "Condition":{"StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com",
   "token.actions.githubusercontent.com:sub":"repo:${REPO}:ref:refs/heads/main"}}}]}
JSON
aws iam create-role --role-name boreal-github-deploy --assume-role-policy-document file://trust-github.json || true

cat > deploy-policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecs:CreateCluster","ecs:RegisterTaskDefinition","ecs:DescribeTaskDefinition",
   "ecs:CreateExpressGatewayService","ecs:UpdateExpressGatewayService","ecs:DescribeExpressGatewayService",
   "ecs:DescribeClusters","ecs:DescribeServices","ecs:UpdateService","ecs:ListServiceDeployments",
   "ecs:DescribeServiceDeployments","ecs:TagResource","ecs:UntagResource"],"Resource":"*"},
 {"Effect":"Allow","Action":"iam:PassRole",
   "Resource":["arn:aws:iam::${ACCOUNT}:role/ecsTaskExecutionRole","arn:aws:iam::${ACCOUNT}:role/ecsInfrastructureRoleForExpressServices"],
   "Condition":{"StringEquals":{"iam:PassedToService":"ecs.amazonaws.com"}}},
 {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:BatchGetImage","ecr:CompleteLayerUpload",
   "ecr:GetDownloadUrlForLayer","ecr:InitiateLayerUpload","ecr:PutImage","ecr:UploadLayerPart"],
   "Resource":["arn:aws:ecr:ca-central-1:${ACCOUNT}:repository/boreal-api","arn:aws:ecr:ca-central-1:${ACCOUNT}:repository/boreal-ui"]},
 {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"}]}
JSON
aws iam put-role-policy --role-name boreal-github-deploy --policy-name deploy --policy-document file://deploy-policy.json
echo DONE
```

If any command prints an error other than "already exists", copy it and send it over before continuing.

## 2. First deployment
GitHub → Actions → **Deploy** → Run workflow (branch `main`). The first run builds and pushes both images,
creates the two ECS Express services (first creation takes several minutes), then runs the smoke test, which
fails the run unless the API reports exactly the deployed commit SHA. The run summary lists the API and UI URLs.
After that, every push to `main` that passes CI deploys automatically.

## 3. Known limits of this deployment
- **The prediction log is a local SQLite file inside the single API task**, so the monitoring history resets
  on each deployment. Run `scripts/simulate_traffic.py --api <url>` after deploying to repopulate it for demos.
  Production would use Amazon RDS for PostgreSQL (the code already supports it through `DATABASE_URL`).
- One task per service (no high availability) to keep cost and state simple.
- Rollback: re-run the Deploy workflow with an earlier commit's workflow run, or update the services to an earlier
  SHA-tagged image in the ECS console.

## 4. Tear-down after the course
ECS console → Clusters → default → Services → delete `boreal-api` and `boreal-ui`; ECR → delete both repositories;
confirm in Billing that the load balancer is gone.
