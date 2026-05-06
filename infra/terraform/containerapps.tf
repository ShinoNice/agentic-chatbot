locals {
  api_app_name     = "ca-${var.name_prefix}-api"
  ui_app_name      = "ca-${var.name_prefix}-ui"
  acr_login_server = "${var.acr_name}.azurecr.io"

  # Internal DNS the UI uses to reach the API inside the ACA env.
  api_internal_url = "http://${local.api_app_name}"

  # Versionless KV secret IDs — ACA resolves the latest version on each
  # replica refresh, mirroring the Bicep keyVaultUrl behavior.
  openai_kv_secret_id    = "${data.azurerm_key_vault.kv.vault_uri}secrets/OPENAI-API-KEY"
  pinecone_kv_secret_id  = "${data.azurerm_key_vault.kv.vault_uri}secrets/PINECONE-API-KEY"
  langsmith_kv_secret_id = "${data.azurerm_key_vault.kv.vault_uri}secrets/LANGSMITH-API-KEY"
}

# ── API container app (internal ingress) ─────────────────────────────────────

resource "azurerm_container_app" "api" {
  name                         = local.api_app_name
  resource_group_name          = data.azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.aca.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.uami.id]
  }

  registry {
    server   = local.acr_login_server
    identity = azurerm_user_assigned_identity.uami.id
  }

  secret {
    name                = "openai-api-key"
    key_vault_secret_id = local.openai_kv_secret_id
    identity            = azurerm_user_assigned_identity.uami.id
  }

  secret {
    name                = "pinecone-api-key"
    key_vault_secret_id = local.pinecone_kv_secret_id
    identity            = azurerm_user_assigned_identity.uami.id
  }

  secret {
    name                = "langsmith-api-key"
    key_vault_secret_id = local.langsmith_kv_secret_id
    identity            = azurerm_user_assigned_identity.uami.id
  }

  ingress {
    external_enabled = false
    target_port      = 8001
    transport        = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name    = "api"
      image   = "${local.acr_login_server}/chatbot-api:${var.image_tag}"
      cpu     = tonumber(var.cpu)
      memory  = var.memory
      command = ["uvicorn"]
      args    = ["src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]

      liveness_probe {
        transport               = "HTTP"
        path                    = "/api/healthz"
        port                    = 8001
        initial_delay           = 30
        interval_seconds        = 30
        failure_count_threshold = 3
      }

      readiness_probe {
        transport               = "HTTP"
        path                    = "/api/readyz"
        port                    = 8001
        interval_seconds        = 10
        failure_count_threshold = 10
      }

      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }
      env {
        name        = "PINECONE_API_KEY"
        secret_name = "pinecone-api-key"
      }
      env {
        name        = "LANGSMITH_API_KEY"
        secret_name = "langsmith-api-key"
      }
      env {
        name  = "LANGSMITH_TRACING"
        value = "true"
      }
      env {
        name  = "LANGSMITH_ENDPOINT"
        value = "https://eu.api.smith.langchain.com"
      }
      env {
        name  = "LANGSMITH_PROJECT"
        value = var.langsmith_project
      }
      env {
        name  = "PYTHONPATH"
        value = "."
      }
      env {
        name  = "ENV"
        value = "production"
      }
      env {
        name  = "DEBUG"
        value = "false"
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.acr_pull,
    azurerm_role_assignment.kv_secrets_user,
  ]
}

# ── UI container app (public ingress) ────────────────────────────────────────
# Angular SPA served by nginx, built from frontend/Dockerfile.
# Listens on 8080. nginx reverse-proxies /api/* to the api container app
# via internal ACA DNS, so API_BASE_URL is empty (= same-origin) and the
# browser never needs to know the api hostname.
# Image entrypoint is nginx — no command/args override here.

resource "azurerm_container_app" "ui" {
  name                         = local.ui_app_name
  resource_group_name          = data.azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.aca.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.uami.id]
  }

  registry {
    server   = local.acr_login_server
    identity = azurerm_user_assigned_identity.uami.id
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name   = "ui"
      image  = "${local.acr_login_server}/chatbot-ui:${var.image_tag}"
      cpu    = tonumber(var.cpu)
      memory = var.memory

      env {
        name = "API_BASE_URL"
        # Empty string = same-origin. nginx /api/ block proxies through to
        # the api container, so the browser does not need a backend URL.
        value = ""
      }
    }
  }

  depends_on = [
    azurerm_role_assignment.acr_pull,
    azurerm_container_app.api,
  ]
}
