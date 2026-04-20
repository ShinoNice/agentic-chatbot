// ─────────────────────────────────────────────────────────────────────────────
// Agentic Chatbot — Azure Container Apps stack
//
// Provisions every piece of infra the live demo needs:
//   - Log Analytics workspace (needed by the ACA env for logs/metrics)
//   - Azure Container Apps environment
//   - User-assigned managed identity (UAMI) shared by both apps
//   - Role assignments: AcrPull on the ACR, KV Secrets User on the KV
//   - Two container apps:
//       * ca-chatbot-api  — internal ingress on 8001
//       * ca-chatbot-ui   — public ingress on 8501
//
// Secrets are pulled from Azure Key Vault via keyVaultUrl references, so
// rotating a secret in KV propagates on the next replica refresh — no more
// copy-paste snapshots on the Container App.
//
// Preconditions (these are NOT created here to keep the IaC idempotent
// against long-lived resources that hold state):
//   - Resource group exists
//   - ACR exists and both images (chatbot-api:<tag>, chatbot-ui:<tag>) are pushed
//   - Key Vault exists with secrets: OPENAI-API-KEY, PINECONE-API-KEY,
//     LANGSMITH-API-KEY  (names match existing prod KV)
// ─────────────────────────────────────────────────────────────────────────────

targetScope = 'resourceGroup'

// ── Parameters ───────────────────────────────────────────────────────────────

@description('Azure region for the ACA environment (North Europe has free-tier quota).')
param location string = 'northeurope'

@description('Short-name prefix for resources (<= 11 chars, lowercase).')
param namePrefix string = 'chatbot'

@description('Name of the existing Azure Container Registry.')
param acrName string = 'acrhybridchatbot'

@description('Resource group holding the ACR (may differ from this RG).')
param acrResourceGroup string = resourceGroup().name

@description('Name of the existing Key Vault that stores LLM/Pinecone/LangSmith secrets.')
param keyVaultName string = 'kv-hybrid-chatbot'

@description('Image tag to deploy for both apps (e.g. "v2").')
param imageTag string = 'v2'

@description('CPU cores per replica (0.25–2.0 in increments of 0.25).')
param cpu string = '0.5'

@description('Memory per replica (e.g. 1Gi, 2Gi).')
param memory string = '1.0Gi'

@description('LangSmith project name for tracing; leave empty to disable.')
param langsmithProject string = 'Agentic_Chatbot'

@description('Public FQDN(s) allowed by the API CORS policy (in addition to localhost).')
param corsExtraOrigins array = []

// ── Derived names ────────────────────────────────────────────────────────────

var logAnalyticsName = 'log-${namePrefix}'
var acaEnvName       = 'cae-${namePrefix}'
var identityName     = 'id-${namePrefix}'
var apiAppName       = 'ca-${namePrefix}-api'
var uiAppName        = 'ca-${namePrefix}-ui'

// Internal DNS the UI uses to reach the API inside the ACA env.
var apiInternalUrl = 'http://${apiAppName}'

// ── Existing resources ───────────────────────────────────────────────────────

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: acrName
  scope: resourceGroup(acrResourceGroup)
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

// ── Observability ────────────────────────────────────────────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── ACA environment ──────────────────────────────────────────────────────────

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ── Identity shared by both apps ─────────────────────────────────────────────

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// Role IDs (stable Azure built-in role definitions).
var acrPullRoleId             = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, 'AcrPull')
  scope: acr
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      acrPullRoleId
    )
  }
}

resource kvSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, uami.id, 'KVSecretsUser')
  scope: kv
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      keyVaultSecretsUserRoleId
    )
  }
}

// ── API container app (internal ingress) ─────────────────────────────────────

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  dependsOn: [acrPull, kvSecretsUser]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 8001
        transport: 'http'
        traffic: [ { weight: 100, latestRevision: true } ]
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
      secrets: [
        {
          name: 'openai-api-key'
          keyVaultUrl: '${kv.properties.vaultUri}secrets/OPENAI-API-KEY'
          identity: uami.id
        }
        {
          name: 'pinecone-api-key'
          keyVaultUrl: '${kv.properties.vaultUri}secrets/PINECONE-API-KEY'
          identity: uami.id
        }
        {
          name: 'langsmith-api-key'
          keyVaultUrl: '${kv.properties.vaultUri}secrets/LANGSMITH-API-KEY'
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/chatbot-api:${imageTag}'
          command: ['uvicorn']
          args: ['src.api.app:app', '--host', '0.0.0.0', '--port', '8001']
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/api/healthz', port: 8001 }
              initialDelaySeconds: 30
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: { path: '/api/readyz', port: 8001 }
              initialDelaySeconds: 45
              periodSeconds: 10
              failureThreshold: 10
            }
          ]
          env: [
            { name: 'OPENAI_API_KEY',    secretRef: 'openai-api-key' }
            { name: 'PINECONE_API_KEY',  secretRef: 'pinecone-api-key' }
            { name: 'LANGSMITH_API_KEY', secretRef: 'langsmith-api-key' }
            { name: 'LANGSMITH_TRACING',  value: 'true' }
            { name: 'LANGSMITH_ENDPOINT', value: 'https://eu.api.smith.langchain.com' }
            { name: 'LANGSMITH_PROJECT',  value: langsmithProject }
            { name: 'PYTHONPATH',         value: '.' }
            { name: 'ENV',                value: 'production' }
            { name: 'DEBUG',              value: 'false' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// ── UI container app (public ingress) ────────────────────────────────────────

resource uiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: uiAppName
  location: location
  dependsOn: [acrPull, apiApp]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
        traffic: [ { weight: 100, latestRevision: true } ]
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: '${acr.properties.loginServer}/chatbot-ui:${imageTag}'
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: [
            { name: 'API_BASE_URL',                value: apiInternalUrl }
            { name: 'STREAMLIT_SERVER_PORT',       value: '8501' }
            { name: 'STREAMLIT_SERVER_ADDRESS',    value: '0.0.0.0' }
            { name: 'STREAMLIT_SERVER_HEADLESS',   value: 'true' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// ── Outputs ──────────────────────────────────────────────────────────────────

output uiFqdn string          = uiApp.properties.configuration.ingress.fqdn
output uiPublicUrl string     = 'https://${uiApp.properties.configuration.ingress.fqdn}'
output apiInternalUrl string  = apiInternalUrl
output managedIdentityId string = uami.id
#disable-next-line outputs-should-not-contain-secrets
output corsHintForSettingsYaml array = union(
  [ 'https://${uiApp.properties.configuration.ingress.fqdn}', 'http://localhost:8501', 'http://localhost:8001' ],
  corsExtraOrigins
)
