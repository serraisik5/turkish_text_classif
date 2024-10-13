FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Make the entrypoint script executable
#RUN chmod +x ./scripts/entrypoint.sh

# entry point script
#ENTRYPOINT ["./scripts/entrypoint.sh"]

CMD ["/bin/bash"]