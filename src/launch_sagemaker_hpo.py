"""
D3 — Launch a SageMaker hyperparameter tuning job for src/train.py
(Bayesian strategy, managed spot instances, region ca-central-1), per spec
3.5.

*** NOT RUN. *** This requires the student's own AWS account, budget alarm,
and SageMaker execution role — none of which exist in the build sandbox
(its AWS_* environment variables belong to the Claude Code cloud runtime,
not the student's account, and must not be used for coursework). Run this
from your own machine/AWS CLI profile after:

  1. `aws configure` (or SSO) with an account that has the $10 budget alarm
     from D1 already in place.
  2. Uploading data/processed/gold_customer_features.csv to an S3 bucket
     you control:
         aws s3 cp data/processed/gold_customer_features.csv \\
             s3://<your-bucket>/boreal-marche/gold_customer_features.csv
  3. Creating (or reusing) a SageMaker execution role with S3 read/write
     and CloudWatch Logs permissions, and putting its ARN below.
  4. `pip install sagemaker boto3`
  5. python src/launch_sagemaker_hpo.py --bucket <your-bucket> --role-arn <arn>

After the job completes, pull results with:
    tuner.analytics().dataframe().to_csv("docs/head_a_sagemaker_hpo_results.csv")
and compare against docs/head_a_tuned_params.json (the local Optuna run) —
same search space, same metric — for the D3 "local-vs-managed comparison
table: best score, wall-clock, cost, and a stated crossover point" that the
rubric asks for the written report, not for this script to compute itself.
"""

import argparse

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, IntegerParameter, HyperparameterTuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket you control")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--max-parallel-jobs", type=int, default=2)
    args = parser.parse_args()

    # Relies on the AWS CLI's configured default region/profile; pass
    # --region above purely for documentation of the required ca-central-1
    # region (set it via `aws configure` or AWS_DEFAULT_REGION).
    session = sagemaker.Session()

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",
        role=args.role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        use_spot_instances=True,          # managed spot, per spec
        max_run=1800,                     # 30 min hard cap — cost discipline
        max_wait=3600,
        sagemaker_session=session,
        hyperparameters={
            # Fixed hyperparameters NOT being tuned go here; the tuned ones
            # are supplied by the HyperparameterTuner below instead.
            "min-samples-split": 2,
        },
        metric_definitions=[
            {"Name": "pr_auc", "Regex": r'"pr_auc":\s*([0-9\.]+)'},
        ],
    )

    hyperparameter_ranges = {
        "n-estimators": IntegerParameter(100, 800),
        "max-depth": IntegerParameter(3, 20),
        "min-samples-leaf": IntegerParameter(1, 30),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="pr_auc",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": "pr_auc", "Regex": r'"pr_auc":\s*([0-9\.]+)'}],
        strategy="Bayesian",              # per spec 3.5
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
    )

    train_input = f"s3://{args.bucket}/boreal-marche/"
    tuner.fit({"train": train_input})

    print("Tuning job submitted:", tuner.latest_tuning_job.name)
    print("Poll status with: tuner.wait() or the SageMaker console.")
    print("After completion: tuner.analytics().dataframe() for the comparison table.")


if __name__ == "__main__":
    main()
