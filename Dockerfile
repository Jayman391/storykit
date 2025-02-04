# syntax=docker/dockerfile:1.2
# Use the official Python 3.11 image as the base image.
FROM python:3.11

# Set the working directory in the container.
WORKDIR /app

# Copy requirements.txt into the container.
COPY requirements.txt .

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the monkey patch script and run it.
COPY monkeypatch_shifterator.py .
RUN python monkeypatch_shifterator.py

# Copy the rest of your application code.
COPY . .

# Expose the port your Dash app will run on (Dash defaults to 8050).
EXPOSE 8050

# --- Clone, Build, and Install babycenterdb ---

# Ensure the GitLab host is known to avoid host key verification errors.
RUN mkdir -p /root/.ssh && ssh-keyscan gitlab.com >> /root/.ssh/known_hosts

# Clone the babycenterdb repository via SSH using BuildKit SSH forwarding.
RUN --mount=type=ssh git clone git@gitlab.com:compstorylab/babycenterdb.git /app/babycenterdb

# Build a wheel for babycenterdb:
# 1. Ensure pip can build wheels.
# 2. Create a directory for the wheel.
# 3. Build the wheel from the cloned repository.
RUN pip install wheel && \
    mkdir -p /app/wheels && \
    cd /app/babycenterdb && \
    python setup.py bdist_wheel --dist-dir /app/wheels

# Install the built wheel file.
RUN pip install /app/wheels/babycenterdb-*.whl

# Define the default command to run your application.
CMD ["python", "app.py"]
