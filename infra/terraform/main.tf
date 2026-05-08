# ─────────────────────────────────────────────────────────────────────────────
# Agentic Chatbot — Azure Container Apps stack
#
# Provisions:
#   - Log Analytics workspace (needed by the ACA env for logs/metrics)
#   - Azure Container Apps environment
#   - User-assigned managed identity (UAMI) shared by both apps
#   - Role assignments: AcrPull on the ACR, KV Secrets User on the KV
#
# Container apps live in containerapps.tf.
#
# Preconditions (referenced as data sources, NOT created here):
#   - Resource group exists
#   - ACR exists and both images are pushed
#   - Key Vault exists with secrets: OPENAI-API-KEY, PINECONE-API-KEY,
#     LANGSMITH-API-KEY
# ─────────────────────────────────────────────────────────────────────────────

# ── Existing resources ───────────────────────────────────────────────────────

data "azurerm_resource_group" "this" {
  name = var.resource_group_name
}

data "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = data.azurerm_resource_group.this.name
}

data "azurerm_key_vault" "kv" {
  name                = var.key_vault_name
  resource_group_name = data.azurerm_resource_group.this.name
}

# ── Observability ────────────────────────────────────────────────────────────

resource "azurerm_log_analytics_workspace" "la" {
  name                = "log-${var.name_prefix}"
  location            = var.location
  resource_group_name = data.azurerm_resource_group.this.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# ── ACA environment ──────────────────────────────────────────────────────────

resource "azurerm_container_app_environment" "aca" {
  name                       = "cae-${var.name_prefix}"
  location                   = var.location
  resource_group_name        = data.azurerm_resource_group.this.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.la.id
}

# ── Identity shared by both apps ─────────────────────────────────────────────

resource "azurerm_user_assigned_identity" "uami" {
  name                = "id-${var.name_prefix}"
  location            = var.location
  resource_group_name = data.azurerm_resource_group.this.name
}

# ── Role assignments ─────────────────────────────────────────────────────────
# Built-in role: AcrPull
resource "azurerm_role_assignment" "acr_pull" {
  scope                = data.azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.uami.principal_id
  principal_type       = "ServicePrincipal"
}

# Built-in role: Key Vault Secrets User
resource "azurerm_role_assignment" "kv_secrets_user" {
  scope                = data.azurerm_key_vault.kv.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.uami.principal_id
  principal_type       = "ServicePrincipal"
}
