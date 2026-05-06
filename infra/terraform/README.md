# Terraform — Agentic Chatbot infra

Mirror translation of `infra/main.bicep` into Terraform. Same resources,
same names, same outputs. The Bicep is kept as the source of truth until
a future cutover commit; this tree exists so we can validate parity.

## Prerequisites

- Terraform >= 1.5
- Azure CLI logged in (`az login`) with rights on the target subscription
- An Azure Storage account to hold the remote state (one-time setup, below)
- The preconditions from the Bicep still apply:
  - Resource group `rg-agentic-chatbot` exists
  - ACR `acrhybridchatbot` exists with `chatbot-api:<tag>` and `chatbot-ui:<tag>`
  - Key Vault `kv-hybrid-chatbot` exists with secrets `OPENAI-API-KEY`,
    `PINECONE-API-KEY`, `LANGSMITH-API-KEY`

## 1. One-time state backend setup

Run once per environment. Storage account names must be globally unique
and lowercase alphanumeric, <= 24 chars.

```bash
LOCATION="northeurope"
TFSTATE_RG="rg-tfstate-agentic-chatbot"
TFSTATE_SA="sttfstateagenticchatbot"   # adjust if taken
TFSTATE_CONTAINER="tfstate"

az group create --name "$TFSTATE_RG" --location "$LOCATION"

az storage account create \
  --name "$TFSTATE_SA" \
  --resource-group "$TFSTATE_RG" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --encryption-services blob \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false

az storage container create \
  --name "$TFSTATE_CONTAINER" \
  --account-name "$TFSTATE_SA" \
  --auth-mode login
```

Save the values for the next step.

## 2. Init

```bash
cd infra/terraform

terraform init \
  -backend-config="resource_group_name=rg-tfstate-agentic-chatbot" \
  -backend-config="storage_account_name=sttfstateagenticchatbot" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=agentic-chatbot.tfstate"
```

Or commit a `prod.tfbackend` file (gitignored) and run
`terraform init -backend-config=prod.tfbackend`.

## 3. Plan

```bash
terraform plan -var "image_tag=$(git rev-parse --short HEAD)"
```

For a parity check against the live Bicep deploy, omit the `-var` and let
the default in `terraform.tfvars` (`image_tag = "v5"`) match what's running
in prod. The plan should show zero resource changes once the import step
below is done.

## 4. Apply

```bash
terraform apply -var "image_tag=$(git rev-parse --short HEAD)"
```

## 5. Importing existing resources (zero-downtime migration from Bicep)

The first `terraform plan` against an empty state will want to recreate
everything. To avoid downtime, import each existing resource into state
before the first apply. Replace `<SUB_ID>` with your subscription ID.

```bash
SUB="<SUB_ID>"
RG="rg-agentic-chatbot"

terraform import azurerm_log_analytics_workspace.la \
  "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.OperationalInsights/workspaces/log-chatbot"

terraform import azurerm_container_app_environment.aca \
  "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/managedEnvironments/cae-chatbot"

terraform import azurerm_user_assigned_identity.uami \
  "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-chatbot"

terraform import azurerm_container_app.api \
  "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/ca-chatbot-api"

terraform import azurerm_container_app.ui \
  "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/ca-chatbot-ui"
```

Role assignments use the assignment GUID, which you can list via
`az role assignment list --assignee <UAMI_PRINCIPAL_ID> --all -o table`.
Then:

```bash
terraform import azurerm_role_assignment.acr_pull \
  "<scope_id_of_acr>/providers/Microsoft.Authorization/roleAssignments/<acr_pull_assignment_guid>"

terraform import azurerm_role_assignment.kv_secrets_user \
  "<scope_id_of_kv>/providers/Microsoft.Authorization/roleAssignments/<kv_secrets_assignment_guid>"
```

After every import, re-run `terraform plan`. The goal is a clean plan
with `No changes.` Anything that wants to drift is something to
investigate before applying.

## CI auth

`providers.tf` sets `use_oidc = true` for GitHub Actions OIDC federation.
The action must export `ARM_SUBSCRIPTION_ID`, `ARM_TENANT_ID`, and
`ARM_CLIENT_ID` — no client secret needed.
