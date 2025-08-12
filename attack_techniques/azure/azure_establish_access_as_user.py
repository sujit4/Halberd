import json
import subprocess
from typing import Any, Dict, Tuple

from core.azure.azure_access import AzureAccess

from ..base_technique import (
    AzureTRMTechnique,
    BaseTechnique,
    ExecutionStatus,
    MitreTechnique,
)
from ..technique_registry import TechniqueRegistry


@TechniqueRegistry.register
class AzureEstablishAccessAsUser(BaseTechnique):
    def __init__(self):
        mitre_techniques = [
            MitreTechnique(
                technique_id="T1078.004",
                technique_name="Valid Accounts",
                tactics=[
                    "Defense Evasion",
                    "Persistence",
                    "Privilege Escalation",
                    "Initial Access",
                ],
                sub_technique_name="Cloud Accounts",
            )
        ]
        azure_trm_technique = [
            AzureTRMTechnique(
                technique_id="AZT201.1",
                technique_name="Valid Credentials",
                tactics=["Initial Access"],
                sub_technique_name="User Account",
            )
        ]
        super().__init__(
            "Establish Access As User",
            "Authenticates to an Azure tenant using username and password credentials. The technique attempts programmatic authentication via the Azure CLI and falls back to interactive browser login if initial authentication fails. This supports scenarios where additional authentication prompts may be required. Successfully authenticated sessions return information about accessible subscriptions including subscription IDs, account details, and tenant associations. Commonly used during initial access after credential theft or phishing attacks.",
            mitre_techniques,
            azure_trm_technique,
        )

    def execute(self, **kwargs: Any) -> Tuple[ExecutionStatus, Dict[str, Any]]:
        self.validate_parameters(kwargs)
        try:
            # input validation
            username: str = kwargs["username"]
            password: str = kwargs["password"]

            if username in ["", None] or password in ["", None]:
                return ExecutionStatus.FAILURE, {
                    "error": {"Error": "Invalid Technique Input"},
                    "message": {"Error": "Invalid Technique Input"},
                }

            # get az full execution path
            az_command = AzureAccess().az_command
            raw_response = subprocess.run(
                [az_command, "login", "-u", username, "-p", password],
                capture_output=True,
            )

            # if login attempt fails, launch interactive login
            if raw_response.returncode == 1:
                raw_response = subprocess.run(
                    [az_command, "login"], capture_output=True
                )

            if raw_response.returncode == 0:
                output = raw_response.stdout
                struc_output = json.loads(output.decode("utf-8"))

                try:
                    output = {}
                    for subscription in struc_output:
                        output[subscription.get("id", "N/A")] = {
                            "subscription_name": subscription.get("name", "N/A"),
                            "subscription_id": subscription.get("id", "N/A"),
                            "home_tenant_id": subscription.get("homeTenantId", "N/A"),
                            "state": subscription.get("state", "N/A"),
                            "identity": subscription.get("user", "N/A").get(
                                "name", "N/A"
                            ),
                            "identity_type": subscription.get("user", "N/A").get(
                                "type", "N/A"
                            ),
                        }
                    return ExecutionStatus.SUCCESS, {
                        "message": "Successfully established access to target Azure tenant",
                        "value": output,
                    }
                except:
                    return ExecutionStatus.PARTIAL_SUCCESS, {
                        "message": "Successfully established access to target Azure tenant",
                        "value": struc_output,
                    }
            else:
                return ExecutionStatus.FAILURE, {
                    "error": str(raw_response.returncode),
                    "message": "Failed to establish access to Azure tenant",
                }
        except Exception as e:
            return ExecutionStatus.FAILURE, {
                "error": str(e),
                "message": "Failed to establish access to Azure tenant",
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "username": {
                "type": "str",
                "required": True,
                "default": None,
                "name": "Username",
                "input_field_type": "text",
            },
            "password": {
                "type": "str",
                "required": True,
                "default": None,
                "name": "Password",
                "input_field_type": "password",
            },
        }
