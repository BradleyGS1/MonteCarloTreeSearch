pipeline {
    agent { label 'agent1' }

    stages {
        stage('Set Up Python') {
            steps {
                // Install Python virtual env using pip
                sh 'python3 -m venv venv'

                // Activate the virtual environment
                sh 'source venv/bin/activate'

                // Upgrade pip inside the virtual environmpent
                sh './venv/bin/pip install --upgrade pip'

                // Install python dependencies using pip
                sh './venv/bin/pip install -r requirements.txt'
            }
        }

        stage('Lint Code') {
            steps {
                // Lint the code using flake8
                sh './venv/bin/pip install flake8'
                sh './venv/bin/flake8 src/'
            }
        }

        stage('Run Tests') {
            steps {
                // Run tests using pytest
                sh './venv/bin/pip install pytest'
                sh './venv/bin/pytest tests/'
            }
        }
    }

    post {
        always {
            // Always clean up environment
            sh 'echo "Cleaning up..."'
            sh 'rm -rf venv'
        }
    }
}
