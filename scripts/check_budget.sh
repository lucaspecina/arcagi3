#!/bin/bash
# Check Azure AI Foundry gpt-5.4 token consumption and estimated cost
# Run periodically during autoresearch to monitor spending

set -e

echo "=== Azure AI Foundry Budget Check ==="
echo "Resource: amalia-resource | Budget: \$70 USD total for gpt-5.4"
echo "Pricing: Input ~\$3/1M | Output ~\$12/1M"
echo ""

# Check if az CLI is available and logged in
if ! command -v az &> /dev/null; then
    echo "ERROR: Azure CLI (az) not installed. Install with: brew install azure-cli"
    echo "Then login with: az login --tenant 23465ce0-8488-41b6-9fa6-fd6b7efaeb44"
    exit 1
fi

RESOURCE="/subscriptions/30dac9f4-4d7c-47af-ae82-7a9a46cb5763/resourceGroups/RG-AmalIA-Dev/providers/Microsoft.CognitiveServices/accounts/amalia-resource"

echo "--- Token consumption (last 7 days, gpt-5.4) ---"
az monitor metrics list \
    --resource "$RESOURCE" \
    --metric "ProcessedPromptTokens" "GeneratedTokens" \
    --interval P1D \
    --start-time "$(date -u -v-7d +%Y-%m-%dT00:00:00Z)" \
    --end-time "$(date -u +%Y-%m-%dT23:59:59Z)" \
    --aggregation Total \
    --filter "ModelDeploymentName eq 'gpt-5.4'" \
    -o table 2>/dev/null || echo "  (Could not fetch metrics — may need: az login --tenant 23465ce0-8488-41b6-9fa6-fd6b7efaeb44)"

echo ""
echo "--- Cost estimate (MonthToDate, RG-AmalIA-Dev) ---"
az rest --method post \
    --uri "https://management.azure.com/subscriptions/30dac9f4-4d7c-47af-ae82-7a9a46cb5763/providers/Microsoft.CostManagement/query?api-version=2023-11-01" \
    --body '{
      "type": "ActualCost",
      "timeframe": "MonthToDate",
      "dataset": {
        "granularity": "None",
        "aggregation": { "totalCost": { "name": "Cost", "function": "Sum" } },
        "filter": { "dimensions": { "name": "ResourceGroup", "operator": "In", "values": ["RG-AmalIA-Dev"] } },
        "grouping": [{ "type": "Dimension", "name": "MeterSubCategory" }]
      }
    }' 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    rows = data.get('properties', {}).get('rows', [])
    total = 0
    for row in rows:
        cost = row[0]
        category = row[1]
        total += cost
        print(f'  {category}: \${cost:.2f}')
    print(f'  ---')
    print(f'  TOTAL: \${total:.2f} / \$70.00 budget')
    pct = total / 70 * 100
    if pct > 85:
        print(f'  ⚠️  WARNING: {pct:.0f}% of budget used! Consider stopping gpt-5.4 usage.')
    elif pct > 60:
        print(f'  ⚠️  CAUTION: {pct:.0f}% of budget used.')
    else:
        print(f'  ✅ OK: {pct:.0f}% of budget used.')
except Exception as e:
    print(f'  (Could not parse cost data: {e})')
" || echo "  (Could not fetch cost data — may need az login)"

echo ""
echo "=== Check complete ==="
