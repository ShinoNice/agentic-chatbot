# Infrastructure as Code

Bicep template for the Azure stack behind the live demo. Recreates the full
Container Apps environment in a single `az deployment group create`.

## What this provisions

- Log Analytics workspace (required by the ACA environment)
- Azure Container Apps environment (`cae-chatbot`)
- User-assigned managed identity (`id-chatbot`) shared by both apps
- Role assignments: `AcrPull` on the ACR, `Key Vault Secrets User` on the KV
- `ca-chatbot-api` — internal ingress on 8001, secrets via KV references
- `ca-chatbot-ui`  — public HTTPS ingress on 8501

## What this does NOT provision (by design)

- The **ACR** — kept out of IaC because it holds images (state).
- The **Key Vault** — kept out of IaC because it holds secrets (state).
- The **resource group** itself — created once, reused forever.

These three are long-lived stores of state; Bicep is for the stateless
compute layer sitting on top of them. If you need a brand-new environment,
pre-create the RG + ACR + KV manually (or add a separate `bootstrap.bicep`),
push your images, and populate the KV secrets, then run this template.

## Preconditions

1. Resource group exists.
2. ACR exists and contains `chatbot-api:<tag>` and `chatbot-ui:<tag>`.
3. Key Vault exists with the secrets `OPENAI-API-KEY`, `PINECONE-API-KEY`,
   `LANGSMITH-API-KEY` (names match the deployed stack).
4. Your user / service principal has `Contributor` on the RG.
5. The user / service principal has `User Access Administrator` so Bicep
   can create the role assignments. (Or assign them manually.)

## Deploy

```bash
az deployment group create \
  --resource-group rg-agentic-chatbot \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json
```

Bump the image tag on a new release:

```bash
az deployment group create \
  --resource-group rg-agentic-chatbot \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json \
  --parameters imageTag=v3
```

## What-if (dry-run)

```bash
az deployment group what-if \
  --resource-group rg-agentic-chatbot \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json
```

## Outputs

- `uiPublicUrl` — the deployed Streamlit URL. Wire into CI or drop into the
  top of the main README.
- `corsHintForSettingsYaml` — the origins the committed
  `config/settings.yaml` should contain for production. Useful when bumping
  the UI FQDN after a region migration.
