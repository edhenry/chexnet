from builtins import range
from datetime import timedelta

import airflow
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.operators import ssh_execute_operator

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
}

dag = DAG(
    dag_id='chexnet_training_sched',
    default_args=args,
    schedule_interval='0 0 * * *',
    dagrun_timeout=timedelta(minutes=60),
)

run_training = BashOperator(
    task_id='run_chexnet_training',
    bash_command='docker run --rm --runtime=nvidia nvidia/cuda:9.0-base nvidia-smi',
    dag=dag
)

run_training >> run_this_last

for i in range(3):
    task = BashOperator(
        task_id='runme_' + str(i),
        bash_command='echo "{{ task_instance_key_str }}" && sleep 1',
        dag=dag,
    )
    task >> run_training

also_run_this = BashOperator(
    task_id='also_run_this',
    bash_command='echo "run_id={{ run_id }} | dag_run={{ dag_run }}',
    dag=dag,
)

also_run_this >> run_this_last

if __name__ == '__main__':
    dag.cli()