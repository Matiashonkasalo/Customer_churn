provider "google" {
  project = var.project_id
  region  = var.region
}

# Service account for Cloud Run
resource "google_service_account" "cloud_run_sa" {
  account_id   = "churn-cloud-run"
  display_name = "Churn Cloud Run service account"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "churn_api" {
  name     = "churn-api"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_sa.email

    containers {
      image = var.docker_image

      ports {
        container_port = 8000
      }
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# Make the service public
resource "google_cloud_run_service_iam_member" "public_access" {
  location = var.region
  service  = google_cloud_run_v2_service.churn_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
