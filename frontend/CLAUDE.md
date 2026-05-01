---
name: frontend-project-structure
description: >
  Scaffold new frontend projects following the exact folder and component
  architecture of this codebase. Use whenever creating a new frontend project,
  adding a new feature, new component, new page, or any new module. Enforces
  the per-component folder conventions (redux, styled, hooks, etc.), naming
  conventions, and top-level src layout.
---

# Frontend Project Structure Skill

> Auto-generated from project scan on 2026-05-01.
> **This skill captures STRUCTURE only — not implementation code.**

---

## Project Overview

| Property | Value |
|---|---|
| Framework | React (CRA/Vite) |
| State Management | Redux Redux-Toolkit  |
| Styling | styled-components Emotion Emotion-styled MUI CSS-Modules(2files)  |
| Naming Convention | PascalCase |

---

## Top-Level Directory Layout

```
src
src/app
src/app/components
src/domain
src/domain/Integrations
src/domain/InventoryAssets
src/domain/InventoryDevices
src/domain/InventoryPrivileges
src/domain/InventoryRoleGroup
src/domain/Workspaces
src/domain/__TMP__
src/domain/aggregationsDevices
src/domain/analysis
src/domain/automation
src/domain/backUp
src/domain/complianceEvidences
src/domain/complianceFrameworks
src/domain/compliancePolicies
src/domain/complianceReports
src/domain/config
src/domain/contacts
src/domain/crownJewels
src/domain/findings
src/domain/flows
src/domain/flows_temporary
src/domain/flows_test
src/domain/help
src/domain/identities
src/domain/inventory
src/domain/inventoryAccounts
src/domain/issues
src/domain/leftSideBar
src/domain/leftSideBarSettings
src/domain/license_page
src/domain/monitoring
src/domain/new_overview
src/domain/overview
src/domain/postureApplication
src/domain/profiles
src/domain/queries
src/domain/settingsProfile
src/domain/shell
src/domain/signRestorePass
src/domain/signUp
src/domain/signin
src/domain/snack
src/domain/support
src/domain/thirdPartyApp
src/domain/topBar
src/shared
```

---

## Component Roots (where components live)

```
src/app/components
src/domain/aggregationsDevices/components
src/domain/aggregationsDevices/pages
src/domain/analysis/components
src/domain/backUp/components
src/domain/complianceEvidences/components
src/domain/complianceFrameworks/components
src/domain/compliancePolicies/components
src/domain/complianceReports/components
src/domain/config/components
src/domain/contacts/components
src/domain/crownJewels/components
src/domain/flows/components
src/domain/flows_temporary/components
src/domain/flows_test/components
src/domain/identities/components
src/domain/Integrations/components
src/domain/Integrations/pages
src/domain/inventoryAccounts/components
src/domain/InventoryAssets/components
src/domain/InventoryDevices/components
src/domain/InventoryPrivileges/components
src/domain/InventoryRoleGroup/components
src/domain/issues/components
src/domain/leftSideBarSettings/components
src/domain/license_page/components
src/domain/monitoring/components
src/domain/new_overview/components
src/domain/overview/components
src/domain/postureApplication/components
src/domain/profiles/components
src/domain/queries/components
src/domain/settingsProfile/components
src/domain/shell/components
src/domain/signin/components
src/domain/signRestorePass/components
src/domain/signUp/components
src/domain/support/components
src/domain/thirdPartyApp/components
src/domain/Workspaces/components
src/domain/Workspaces/pages
src/domain/__TMP__/components
src/shared/components
the-golden-pyramid/src/components
```

---

## Per-Component Sub-Folder Conventions

```
ComponentName/
  ├── hooks/   # found in 44 component(s)
  ├── api/   # found in 27 component(s)
  ├── styled/   # found in 25 component(s)
  ├── store/   # found in 25 component(s)
  ├── helpers/   # found in 10 component(s)
  ├── styles/   # found in 5 component(s)
  ├── utils/   # found in 1 component(s)
  ├── types/   # found in 1 component(s)
  ├── services/   # found in 1 component(s)
  ├── redux/   # found in 1 component(s)
  ├── queries/   # found in 1 component(s)
```

### What goes in each folder

- **hooks/** — Custom React hooks that encapsulate this component's logic.
- **api/** — API calls, service functions, or data-fetching queries.
- **styled/** — All styled-components / CSS / emotion / SASS files. Zero raw style in JSX.
- **store/** — Redux slice, selectors, thunks, and store wiring for this component.
- **helpers/** — Pure utility functions. No React imports allowed here.
- **styles/** — All styled-components / CSS / emotion / SASS files. Zero raw style in JSX.
- **utils/** — Pure utility functions. No React imports allowed here.
- **types/** — TypeScript types and interfaces for this component domain.
- **services/** — API calls, service functions, or data-fetching queries.
- **redux/** — Redux slice, selectors, thunks, and store wiring for this component.
- **queries/** — API calls, service functions, or data-fetching queries.


---

## Concrete Component Examples (sampled from this project)

### Component: `RightSidePanel`
```
RightSidePanel/

  src/domain/aggregationsDevices/components/RightSidePanel/index.tsx
```
### Component: `extended`
```
extended/

  src/domain/aggregationsDevices/pages/extended/index.tsx
```

### Component: `mainPage`
```
mainPage/

  src/domain/aggregationsDevices/pages/mainPage/index.tsx
```
### Component: `styled`
```
styled/

  src/domain/backUp/components/styled/index.tsx
```

### Component: `table`
```
table/

  src/domain/backUp/components/table/index.tsx
```
### Component: `AutoAssetEvidenceRow`
```
AutoAssetEvidenceRow/

  src/domain/complianceEvidences/components/AutoAssetEvidenceRow/index.tsx
```

### Component: `EvidenceSoloRow`
```
EvidenceSoloRow/

  src/domain/complianceEvidences/components/EvidenceSoloRow/index.tsx
```

### Component: `ManualFileEvidenceRow`
```
ManualFileEvidenceRow/

  src/domain/complianceEvidences/components/ManualFileEvidenceRow/index.tsx
```

### Component: `TableAssetDialog`
```
TableAssetDialog/

  src/domain/complianceEvidences/components/TableAssetDialog/index.tsx
```
### Component: `AutomationAssetRow`
```
AutomationAssetRow/

  src/domain/complianceFrameworks/components/AutomationAssetRow/index.tsx
```

### Component: `ControlPage`
```
ControlPage/

  src/domain/complianceFrameworks/components/ControlPage/index.tsx
```

### Component: `Dialog`
```
Dialog/
  src/domain/complianceFrameworks/components/Dialog/AddCustomTextDialog
  src/domain/complianceFrameworks/components/Dialog/AssignPolicyDialog
  src/domain/complianceFrameworks/components/Dialog/FrameworkStoreDialog
  src/domain/complianceFrameworks/components/Dialog/FrameworkStoreDialog/DialogMainTable
  src/domain/complianceFrameworks/components/Dialog/ViewPolicyDialog
  src/domain/complianceFrameworks/components/Dialog/AddCustomTextDialog/index.tsx
  src/domain/complianceFrameworks/components/Dialog/AssignPolicyDialog/index.tsx
  src/domain/complianceFrameworks/components/Dialog/FrameworkStoreDialog/index.tsx
  src/domain/complianceFrameworks/components/Dialog/FrameworkStoreDialog/DialogMainTable/index.tsx
  src/domain/complianceFrameworks/components/Dialog/ViewPolicyDialog/index.tsx
```

### Component: `EvidenceRow`
```
EvidenceRow/

  src/domain/complianceFrameworks/components/EvidenceRow/index.tsx
```

### Component: `FrameworkControlManualEvidenceRow`
```
FrameworkControlManualEvidenceRow/

  src/domain/complianceFrameworks/components/FrameworkControlManualEvidenceRow/index.tsx
```
### Component: `PolicyCreateDialog`
```
PolicyCreateDialog/

  src/domain/compliancePolicies/components/PolicyCreateDialog/index.tsx
```

### Component: `PolicyRow`
```
PolicyRow/

  src/domain/compliancePolicies/components/PolicyRow/index.tsx
```

### Component: `PolicyStoreDialog`
```
PolicyStoreDialog/

  src/domain/compliancePolicies/components/PolicyStoreDialog/index.tsx
```

### Component: `PolicyView`
```
PolicyView/

  src/domain/compliancePolicies/components/PolicyView/index.tsx
```

### Component: `styles`
```
styles/

  src/domain/compliancePolicies/components/styles/index.tsx
```
### Component: `Dialog`
```
Dialog/
  src/domain/complianceReports/components/Dialog/CreateReportDialog
  src/domain/complianceReports/components/Dialog/PreviewReportDialog
  src/domain/complianceReports/components/Dialog/ShareUrlDialog
  src/domain/complianceReports/components/Dialog/triggerRepeat
  src/domain/complianceReports/components/Dialog/CreateReportDialog/index.tsx
  src/domain/complianceReports/components/Dialog/PreviewReportDialog/index.tsx
  src/domain/complianceReports/components/Dialog/ShareUrlDialog/index.tsx
  src/domain/complianceReports/components/Dialog/triggerRepeat/index.tsx
```

### Component: `ReportAuditTable`
```
ReportAuditTable/

  src/domain/complianceReports/components/ReportAuditTable/index.tsx
```

### Component: `ReportCircleCard`
```
ReportCircleCard/

  src/domain/complianceReports/components/ReportCircleCard/index.tsx
```

### Component: `ReportControlsBlock`
```
ReportControlsBlock/

  src/domain/complianceReports/components/ReportControlsBlock/index.tsx
```

### Component: `ReportEvidenceTable`
```
ReportEvidenceTable/

  src/domain/complianceReports/components/ReportEvidenceTable/index.tsx
```
### Component: `AddContactRightSideBar`
```
AddContactRightSideBar/

  src/domain/contacts/components/AddContactRightSideBar/index.tsx
```

### Component: `AddTeamRightSideBar`
```
AddTeamRightSideBar/

  src/domain/contacts/components/AddTeamRightSideBar/index.tsx
```

### Component: `DeleteTeamDialog`
```
DeleteTeamDialog/

  src/domain/contacts/components/DeleteTeamDialog/index.tsx
```

### Component: `EditTeamDialog`
```
EditTeamDialog/

  src/domain/contacts/components/EditTeamDialog/index.tsx
```

### Component: `addUserDialog`
```
addUserDialog/

  src/domain/contacts/components/addUserDialog/index.tsx
```
### Component: `CUComponents`
```
CUComponents/
  src/domain/crownJewels/components/CUComponents/AddApplication
  src/domain/crownJewels/components/CUComponents/AddApplication/Assets
  src/domain/crownJewels/components/CUComponents/AddApplication/Chip
  src/domain/crownJewels/components/CUComponents/AddApplication/GroupRole
  src/domain/crownJewels/components/CUComponents/AddApplication/Privilege
  src/domain/crownJewels/components/CUComponents/ApplicationList
  src/domain/crownJewels/components/CUComponents/CJHeader
  src/domain/crownJewels/components/CUComponents/Inventories
  src/domain/crownJewels/components/CUComponents/Inventories/Panels
  src/domain/crownJewels/components/CUComponents/Inventories/TabsHeader
  src/domain/crownJewels/components/CUComponents/AddApplication/index.tsx
  src/domain/crownJewels/components/CUComponents/AddApplication/Assets/index.tsx
  src/domain/crownJewels/components/CUComponents/AddApplication/Chip/index.tsx
  src/domain/crownJewels/components/CUComponents/AddApplication/GroupRole/index.tsx
  src/domain/crownJewels/components/CUComponents/AddApplication/Privilege/index.tsx
  src/domain/crownJewels/components/CUComponents/ApplicationList/index.tsx
  src/domain/crownJewels/components/CUComponents/CJHeader/index.tsx
  src/domain/crownJewels/components/CUComponents/Inventories/index.tsx
  src/domain/crownJewels/components/CUComponents/Inventories/Panels/index.tsx
  src/domain/crownJewels/components/CUComponents/Inventories/TabsHeader/index.tsx
```

### Component: `CrownJewelsAdd`
```
CrownJewelsAdd/

  src/domain/crownJewels/components/CrownJewelsAdd/index.tsx
```

### Component: `CrownJewelsEdit`
```
CrownJewelsEdit/

  src/domain/crownJewels/components/CrownJewelsEdit/index.tsx
```

### Component: `DeleteDialog`
```
DeleteDialog/

  src/domain/crownJewels/components/DeleteDialog/index.tsx
```

### Component: `MenuEditRemoveApplication`
```
MenuEditRemoveApplication/

  src/domain/crownJewels/components/MenuEditRemoveApplication/index.tsx
```
### Component: `AutoZoom`
```
AutoZoom/

  src/domain/flows/components/AutoZoom/index.tsx
```

### Component: `ColorMarker`
```
ColorMarker/

  src/domain/flows/components/ColorMarker/index.tsx
```

### Component: `Contexts`
```
Contexts/
  src/domain/flows/components/Contexts/FunctionContext
  src/domain/flows/components/Contexts/FunctionContext/index.tsx
```

### Component: `ExpandedBuilderHeader`
```
ExpandedBuilderHeader/

  src/domain/flows/components/ExpandedBuilderHeader/index.tsx
```

### Component: `ExpandedFlowBuilder`
```
ExpandedFlowBuilder/

  src/domain/flows/components/ExpandedFlowBuilder/index.tsx
```
### Component: `styled`
```
styled/

  src/domain/flows_temporary/components/styled/index.ts
```
### Component: `automationHistory`
```
automationHistory/
  src/domain/flows_test/components/automationHistory/table
  src/domain/flows_test/components/automationHistory/index.tsx
  src/domain/flows_test/components/automationHistory/table/index.tsx
```

### Component: `automationSelectionBar`
```
automationSelectionBar/
  src/domain/flows_test/components/automationSelectionBar/addAutomationCard
  src/domain/flows_test/components/automationSelectionBar/automationCard
  src/domain/flows_test/components/automationSelectionBar/index.tsx
  src/domain/flows_test/components/automationSelectionBar/addAutomationCard/index.tsx
  src/domain/flows_test/components/automationSelectionBar/automationCard/index.tsx
```

### Component: `automationTopBar`
```
automationTopBar/

  src/domain/flows_test/components/automationTopBar/index.tsx
```

### Component: `customEdges`
```
customEdges/
  src/domain/flows_test/components/customEdges/plusEdge
  src/domain/flows_test/components/customEdges/plusEdge/index.tsx
```

### Component: `customNodes`
```
customNodes/
  src/domain/flows_test/components/customNodes/dateNode
  src/domain/flows_test/components/customNodes/docChildNode
  src/domain/flows_test/components/customNodes/docNode
  src/domain/flows_test/components/customNodes/emailNode
  src/domain/flows_test/components/customNodes/finishNode
  src/domain/flows_test/components/customNodes/plusNode
  src/domain/flows_test/components/customNodes/repeatNode
  src/domain/flows_test/components/customNodes/saveChildNode
  src/domain/flows_test/components/customNodes/saveNode
  src/domain/flows_test/components/customNodes/stageNode
  src/domain/flows_test/components/customNodes/startNode
  src/domain/flows_test/components/customNodes/triggerNode
  src/domain/flows_test/components/customNodes/dateNode/index.tsx
  src/domain/flows_test/components/customNodes/docChildNode/index.tsx
  src/domain/flows_test/components/customNodes/docNode/index.tsx
  src/domain/flows_test/components/customNodes/emailNode/index.tsx
  src/domain/flows_test/components/customNodes/finishNode/index.tsx
  src/domain/flows_test/components/customNodes/plusNode/index.tsx
  src/domain/flows_test/components/customNodes/repeatNode/index.tsx
  src/domain/flows_test/components/customNodes/saveChildNode/index.tsx
  src/domain/flows_test/components/customNodes/saveNode/index.tsx
  src/domain/flows_test/components/customNodes/stageNode/index.tsx
  src/domain/flows_test/components/customNodes/startNode/index.tsx
  src/domain/flows_test/components/customNodes/triggerNode/index.tsx
```
### Component: `Accounts`
```
Accounts/

  src/domain/identities/components/Accounts/index.tsx
```

### Component: `AccountsRightSideAccount`
```
AccountsRightSideAccount/

  src/domain/identities/components/AccountsRightSideAccount/index.tsx
```

### Component: `AccountsRightSideIdentity`
```
AccountsRightSideIdentity/

  src/domain/identities/components/AccountsRightSideIdentity/index.tsx
```

### Component: `AccountsRightSideOffboard`
```
AccountsRightSideOffboard/

  src/domain/identities/components/AccountsRightSideOffboard/index.tsx
```

### Component: `AccountsTable`
```
AccountsTable/

  src/domain/identities/components/AccountsTable/index.tsx
```
### Component: `Card`
```
Card/

  src/domain/Integrations/components/Card/index.tsx
```

### Component: `DeleteDialog`
```
DeleteDialog/

  src/domain/Integrations/components/DeleteDialog/index.tsx
```

### Component: `GDrive`
```
GDrive/
  src/domain/Integrations/components/GDrive/Instructions
  src/domain/Integrations/components/GDrive/IntegrationDetails
  src/domain/Integrations/components/GDrive/Instructions/index.tsx
  src/domain/Integrations/components/GDrive/IntegrationDetails/index.tsx
```

### Component: `GSuite`
```
GSuite/
  src/domain/Integrations/components/GSuite/Instructions
  src/domain/Integrations/components/GSuite/IntegrationDetails
  src/domain/Integrations/components/GSuite/IntegrationDetails/helpers
  src/domain/Integrations/components/GSuite/Instructions/index.tsx
  src/domain/Integrations/components/GSuite/IntegrationDetails/index.tsx
  src/domain/Integrations/components/GSuite/IntegrationDetails/helpers/index.ts
```

### Component: `Github`
```
Github/
  src/domain/Integrations/components/Github/Instructions
  src/domain/Integrations/components/Github/IntegrationDetails
  src/domain/Integrations/components/Github/Instructions/index.tsx
  src/domain/Integrations/components/Github/IntegrationDetails/index.tsx
```
### Component: `gdrive`
```
gdrive/

  src/domain/Integrations/pages/gdrive/index.tsx
```

### Component: `github`
```
github/

  src/domain/Integrations/pages/github/index.tsx
```

### Component: `gsuite`
```
gsuite/

  src/domain/Integrations/pages/gsuite/index.tsx
```

### Component: `main`
```
main/

  src/domain/Integrations/pages/main/index.tsx
```

### Component: `microsoft`
```
microsoft/

  src/domain/Integrations/pages/microsoft/index.tsx
```
### Component: `AccountsMainTable`
```
AccountsMainTable/

  src/domain/inventoryAccounts/components/AccountsMainTable/index.tsx
```

### Component: `AccountsPage`
```
AccountsPage/

  src/domain/inventoryAccounts/components/AccountsPage/index.tsx
```

### Component: `IdentitySelectSideBar`
```
IdentitySelectSideBar/

  src/domain/inventoryAccounts/components/IdentitySelectSideBar/index.tsx
```

### Component: `RightSidePanel`
```
RightSidePanel/

  src/domain/inventoryAccounts/components/RightSidePanel/index.tsx
```

### Component: `styles`
```
styles/

  src/domain/inventoryAccounts/components/styles/index.tsx
```
### Component: `MainTableAssets`
```
MainTableAssets/

  src/domain/InventoryAssets/components/MainTableAssets/index.tsx
```

### Component: `SideBarInfo`
```
SideBarInfo/

  src/domain/InventoryAssets/components/SideBarInfo/assetsSideBarInfo.test.tsx
  src/domain/InventoryAssets/components/SideBarInfo/index.tsx
```

### Component: `Table`
```
Table/

  src/domain/InventoryAssets/components/Table/assetsTable.test.tsx
  src/domain/InventoryAssets/components/Table/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/InventoryAssets/components/styled/index.tsx
```
### Component: `RightSidePanel`
```
RightSidePanel/

  src/domain/InventoryDevices/components/RightSidePanel/index.tsx
```
### Component: `MainTablePrivileges`
```
MainTablePrivileges/

  src/domain/InventoryPrivileges/components/MainTablePrivileges/index.tsx
```

### Component: `SideBarInfo`
```
SideBarInfo/

  src/domain/InventoryPrivileges/components/SideBarInfo/index.tsx
  src/domain/InventoryPrivileges/components/SideBarInfo/privilegesSideBarInfo.test.tsx
```

### Component: `Table`
```
Table/

  src/domain/InventoryPrivileges/components/Table/index.tsx
  src/domain/InventoryPrivileges/components/Table/privilegesTable.test.tsx
```

### Component: `styled`
```
styled/

  src/domain/InventoryPrivileges/components/styled/index.tsx
```
### Component: `MainTableRoleGroup`
```
MainTableRoleGroup/

  src/domain/InventoryRoleGroup/components/MainTableRoleGroup/index.tsx
```

### Component: `SideBarInfo`
```
SideBarInfo/

  src/domain/InventoryRoleGroup/components/SideBarInfo/index.tsx
  src/domain/InventoryRoleGroup/components/SideBarInfo/roleGroupsSideBarInfo.test.tsx
```

### Component: `Table`
```
Table/

  src/domain/InventoryRoleGroup/components/Table/index.tsx
  src/domain/InventoryRoleGroup/components/Table/roleGroupsTable.test.tsx
```

### Component: `styled`
```
styled/

  src/domain/InventoryRoleGroup/components/styled/index.tsx
```
### Component: `AllIssues`
```
AllIssues/

  src/domain/issues/components/AllIssues/index.tsx
```

### Component: `Dialog`
```
Dialog/
  src/domain/issues/components/Dialog/ChangeAssigneeDialog
  src/domain/issues/components/Dialog/Dialog
  src/domain/issues/components/Dialog/types.ts
  src/domain/issues/components/Dialog/ChangeAssigneeDialog/index.tsx
  src/domain/issues/components/Dialog/Dialog/index.tsx
```

### Component: `Header`
```
Header/

  src/domain/issues/components/Header/index.tsx
```

### Component: `IssuesTable`
```
IssuesTable/

  src/domain/issues/components/IssuesTable/index.tsx
```

### Component: `RightSideContainer`
```
RightSideContainer/

  src/domain/issues/components/RightSideContainer/index.tsx
```
### Component: `styled`
```
styled/

  src/domain/leftSideBarSettings/components/styled/index.ts
```
### Component: `IL_License`
```
IL_License/

  src/domain/license_page/components/IL_License/IL.pdf
  src/domain/license_page/components/IL_License/license.jsx
```
### Component: `ButtonPanelWY`
```
ButtonPanelWY/
  src/domain/monitoring/components/ButtonPanelWY/WhiteButton
  src/domain/monitoring/components/ButtonPanelWY/index.tsx
  src/domain/monitoring/components/ButtonPanelWY/WhiteButton/index.tsx
```

### Component: `MonitoringItem`
```
MonitoringItem/
  src/domain/monitoring/components/MonitoringItem/Notification
  src/domain/monitoring/components/MonitoringItem/Notification/NotificationAssignee
  src/domain/monitoring/components/MonitoringItem/Notification/NotificationRecipients
  src/domain/monitoring/components/MonitoringItem/NotificationRow
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/BodyRow
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/Changeable
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/Dialog
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/hooks
  src/domain/monitoring/components/MonitoringItem/Notification/index.tsx
  src/domain/monitoring/components/MonitoringItem/Notification/NotificationAssignee/index.tsx
  src/domain/monitoring/components/MonitoringItem/Notification/NotificationRecipients/index.tsx
  src/domain/monitoring/components/MonitoringItem/NotificationRow/index.tsx
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/index.tsx
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/BodyRow/index.tsx
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/Changeable/index.tsx
  src/domain/monitoring/components/MonitoringItem/SOURCE_CODE_ADMINS_NOT_IN_PROFILE/Dialog/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/monitoring/components/styled/index.tsx
```
### Component: `Charts`
```
Charts/
  src/domain/new_overview/components/Charts/CustomChart
  src/domain/new_overview/components/Charts/index.tsx
  src/domain/new_overview/components/Charts/CustomChart/index.tsx
```

### Component: `FrameworkOverview`
```
FrameworkOverview/

  src/domain/new_overview/components/FrameworkOverview/index.tsx
```

### Component: `IssuesTable`
```
IssuesTable/

  src/domain/new_overview/components/IssuesTable/index.tsx
```

### Component: `OldOverviewPage`
```
OldOverviewPage/

  src/domain/new_overview/components/OldOverviewPage/index.tsx
```

### Component: `RiskData`
```
RiskData/

  src/domain/new_overview/components/RiskData/index.tsx
```
### Component: `SecurityRiskView`
```
SecurityRiskView/

  src/domain/overview/components/SecurityRiskView/CircleBar.tsx
```
### Component: `Pagination`
```
Pagination/

  src/domain/postureApplication/components/Pagination/index.tsx
```

### Component: `Posture`
```
Posture/
  src/domain/postureApplication/components/Posture/Details
  src/domain/postureApplication/components/Posture/Details/Certifications
  src/domain/postureApplication/components/Posture/Details/Descriptions
  src/domain/postureApplication/components/Posture/Details/History
  src/domain/postureApplication/components/Posture/Details/Owner
  src/domain/postureApplication/components/Posture/Details/Statistics
  src/domain/postureApplication/components/Posture/Details/Status
  src/domain/postureApplication/components/Posture/Details/Title
  src/domain/postureApplication/components/Posture/FilterBlock
  src/domain/postureApplication/components/Posture/PostureHeader
  src/domain/postureApplication/components/Posture/Switcher
  src/domain/postureApplication/components/Posture/Switcher/Switch
  src/domain/postureApplication/components/Posture/Tabulator
  src/domain/postureApplication/components/Posture/Tabulator/Panels
  src/domain/postureApplication/components/Posture/Tabulator/Tabs
  src/domain/postureApplication/components/Posture/index.tsx
  src/domain/postureApplication/components/Posture/Details/index.tsx
  src/domain/postureApplication/components/Posture/Details/Certifications/index.tsx
  src/domain/postureApplication/components/Posture/Details/Descriptions/index.tsx
  src/domain/postureApplication/components/Posture/Details/History/index.tsx
  src/domain/postureApplication/components/Posture/Details/Owner/index.tsx
  src/domain/postureApplication/components/Posture/Details/Statistics/index.tsx
  src/domain/postureApplication/components/Posture/Details/Status/index.tsx
  src/domain/postureApplication/components/Posture/Details/Title/index.tsx
  src/domain/postureApplication/components/Posture/FilterBlock/index.tsx
  src/domain/postureApplication/components/Posture/PostureHeader/index.tsx
  src/domain/postureApplication/components/Posture/Switcher/index.tsx
  src/domain/postureApplication/components/Posture/Switcher/Switch/index.tsx
  src/domain/postureApplication/components/Posture/Tabulator/index.tsx
  src/domain/postureApplication/components/Posture/Tabulator/Panels/index.tsx
  src/domain/postureApplication/components/Posture/Tabulator/Tabs/index.tsx
```

### Component: `Postures`
```
Postures/
  src/domain/postureApplication/components/Postures/FilterBlock
  src/domain/postureApplication/components/Postures/PostureCard
  src/domain/postureApplication/components/Postures/PostureCard/PostureCardChart
  src/domain/postureApplication/components/Postures/PostureHeader
  src/domain/postureApplication/components/Postures/FilterBlock/index.tsx
  src/domain/postureApplication/components/Postures/PostureCard/index.tsx
  src/domain/postureApplication/components/Postures/PostureCard/PostureCardChart/index.tsx
  src/domain/postureApplication/components/Postures/PostureHeader/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/postureApplication/components/styled/index.ts
```
### Component: `AddApplicationRightSideBar`
```
AddApplicationRightSideBar/

  src/domain/profiles/components/AddApplicationRightSideBar/index.tsx
```

### Component: `AddProviderRightSideBar`
```
AddProviderRightSideBar/

  src/domain/profiles/components/AddProviderRightSideBar/index.tsx
```

### Component: `AffectedAccountsRightSideBar`
```
AffectedAccountsRightSideBar/
  src/domain/profiles/components/AffectedAccountsRightSideBar/AffectedAccountList
  src/domain/profiles/components/AffectedAccountsRightSideBar/AffectedAccountList/AffectedAccountItem
  src/domain/profiles/components/AffectedAccountsRightSideBar/Checked
  src/domain/profiles/components/AffectedAccountsRightSideBar/index.tsx
  src/domain/profiles/components/AffectedAccountsRightSideBar/AffectedAccountList/index.tsx
  src/domain/profiles/components/AffectedAccountsRightSideBar/AffectedAccountList/AffectedAccountItem/index.tsx
  src/domain/profiles/components/AffectedAccountsRightSideBar/Checked/index.tsx
  src/domain/profiles/components/AffectedAccountsRightSideBar/Checked/item.tsx
```

### Component: `ApplicationDetailsRightSideBar`
```
ApplicationDetailsRightSideBar/

  src/domain/profiles/components/ApplicationDetailsRightSideBar/index.tsx
```

### Component: `ApplicationShrinkItem`
```
ApplicationShrinkItem/

  src/domain/profiles/components/ApplicationShrinkItem/index.tsx
```
### Component: `TableHeaderItem`
```
TableHeaderItem/

  src/domain/queries/components/TableHeaderItem/index.tsx
```

### Component: `mainQueryPage`
```
mainQueryPage/
  src/domain/queries/components/mainQueryPage/ButtonPanels
  src/domain/queries/components/mainQueryPage/header
  src/domain/queries/components/mainQueryPage/linked
  src/domain/queries/components/mainQueryPage/queryDataPicker
  src/domain/queries/components/mainQueryPage/queryField
  src/domain/queries/components/mainQueryPage/queryWizard
  src/domain/queries/components/mainQueryPage/radioButton
  src/domain/queries/components/mainQueryPage/TableQueryResult
  src/domain/queries/components/mainQueryPage/toolBar
  src/domain/queries/components/mainQueryPage/whiteButton
  src/domain/queries/components/mainQueryPage/wizard
  src/domain/queries/components/mainQueryPage/index.tsx
  src/domain/queries/components/mainQueryPage/ButtonPanels/index.tsx
  src/domain/queries/components/mainQueryPage/header/index.tsx
  src/domain/queries/components/mainQueryPage/linked/index.tsx
  src/domain/queries/components/mainQueryPage/queryDataPicker/index.tsx
  src/domain/queries/components/mainQueryPage/queryDataPicker/queryDataPickerItem.tsx
  src/domain/queries/components/mainQueryPage/queryField/index.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/autocompleteItem.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/index.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/inputItem.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/save.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/saveAutocompleteElement.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/saveElement.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/saveElementWrapper.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/saveInputElement.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/saveSelectElement.tsx
  src/domain/queries/components/mainQueryPage/queryWizard/selectItem.tsx
  src/domain/queries/components/mainQueryPage/radioButton/index.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/dateCell.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/index.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/regularCell.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/tableHeaderQuery.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/tableQueryData.tsx
  src/domain/queries/components/mainQueryPage/TableQueryResult/tableRowQuery.tsx
  src/domain/queries/components/mainQueryPage/toolBar/index.tsx
  src/domain/queries/components/mainQueryPage/whiteButton/index.tsx
  src/domain/queries/components/mainQueryPage/wizard/boxBracket.tsx
  src/domain/queries/components/mainQueryPage/wizard/boxNot.tsx
  src/domain/queries/components/mainQueryPage/wizard/QueryAll.tsx
  src/domain/queries/components/mainQueryPage/wizard/queryApplication.tsx
  src/domain/queries/components/mainQueryPage/wizard/queryElement.tsx
  src/domain/queries/components/mainQueryPage/wizard/queryExist.tsx
  src/domain/queries/components/mainQueryPage/wizard/queryPrivileged.tsx
```

### Component: `pagination`
```
pagination/

  src/domain/queries/components/pagination/index.tsx
```

### Component: `savedQueriesPage`
```
savedQueriesPage/
  src/domain/queries/components/savedQueriesPage/header
  src/domain/queries/components/savedQueriesPage/searchField
  src/domain/queries/components/savedQueriesPage/selectTags
  src/domain/queries/components/savedQueriesPage/switcher
  src/domain/queries/components/savedQueriesPage/TableQuerySaved
  src/domain/queries/components/savedQueriesPage/index.tsx
  src/domain/queries/components/savedQueriesPage/header/headerHead.tsx
  src/domain/queries/components/savedQueriesPage/header/index.tsx
  src/domain/queries/components/savedQueriesPage/searchField/index.js
  src/domain/queries/components/savedQueriesPage/selectTags/index.tsx
  src/domain/queries/components/savedQueriesPage/switcher/index.tsx
  src/domain/queries/components/savedQueriesPage/switcher/switcher.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/index.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/regularCell.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/tableHeaderSavedQuery.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/tableRowSavedQuery.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/tableSavedQueryData.tsx
  src/domain/queries/components/savedQueriesPage/TableQuerySaved/tagCell.tsx
```

### Component: `styled`
```
styled/

  src/domain/queries/components/styled/index.ts
```
### Component: `ChangeName`
```
ChangeName/

  src/domain/settingsProfile/components/ChangeName/index.tsx
```

### Component: `ChangePassword`
```
ChangePassword/

  src/domain/settingsProfile/components/ChangePassword/index.tsx
```

### Component: `SettingsProfileCard`
```
SettingsProfileCard/

  src/domain/settingsProfile/components/SettingsProfileCard/index.tsx
```
### Component: `styled`
```
styled/

  src/domain/shell/components/styled/index.ts
```
### Component: `ContactDialog`
```
ContactDialog/

  src/domain/signin/components/ContactDialog/index.tsx
```

### Component: `LeftSideBar`
```
LeftSideBar/

  src/domain/signin/components/LeftSideBar/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/signin/components/styled/index.tsx
```
### Component: `LeftSideBar`
```
LeftSideBar/

  src/domain/signUp/components/LeftSideBar/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/signUp/components/styled/index.tsx
```
### Component: `styled`
```
styled/

  src/domain/support/components/styled/index.ts
```
### Component: `Pagination`
```
Pagination/

  src/domain/thirdPartyApp/components/Pagination/index.tsx
```

### Component: `application`
```
application/
  src/domain/thirdPartyApp/components/application/Certifications
  src/domain/thirdPartyApp/components/application/Certifications/Item
  src/domain/thirdPartyApp/components/application/DiscoverySources
  src/domain/thirdPartyApp/components/application/DiscoveryStatus
  src/domain/thirdPartyApp/components/application/sso
  src/domain/thirdPartyApp/components/application/sso/Details
  src/domain/thirdPartyApp/components/application/sso/FilterBlock
  src/domain/thirdPartyApp/components/application/Tabulator
  src/domain/thirdPartyApp/components/application/Tabulator/Panels
  src/domain/thirdPartyApp/components/application/Tabulator/Tabs
  src/domain/thirdPartyApp/components/application/tokens
  src/domain/thirdPartyApp/components/application/tokens/Details
  src/domain/thirdPartyApp/components/application/tokens/FilterBlock
  src/domain/thirdPartyApp/components/application/tokens/more
  src/domain/thirdPartyApp/components/application/Certifications/index.tsx
  src/domain/thirdPartyApp/components/application/Certifications/Item/index.tsx
  src/domain/thirdPartyApp/components/application/DiscoverySources/index.tsx
  src/domain/thirdPartyApp/components/application/DiscoveryStatus/index.tsx
  src/domain/thirdPartyApp/components/application/sso/index.tsx
  src/domain/thirdPartyApp/components/application/sso/Details/index.tsx
  src/domain/thirdPartyApp/components/application/sso/FilterBlock/index.tsx
  src/domain/thirdPartyApp/components/application/Tabulator/index.tsx
  src/domain/thirdPartyApp/components/application/Tabulator/Panels/index.tsx
  src/domain/thirdPartyApp/components/application/Tabulator/Tabs/index.tsx
  src/domain/thirdPartyApp/components/application/tokens/index.tsx
  src/domain/thirdPartyApp/components/application/tokens/Details/index.tsx
  src/domain/thirdPartyApp/components/application/tokens/FilterBlock/index.tsx
  src/domain/thirdPartyApp/components/application/tokens/more/index.tsx
```

### Component: `applications`
```
applications/
  src/domain/thirdPartyApp/components/applications/AppCard
  src/domain/thirdPartyApp/components/applications/AppCard/Accounts
  src/domain/thirdPartyApp/components/applications/AppCard/Discovery
  src/domain/thirdPartyApp/components/applications/CardView
  src/domain/thirdPartyApp/components/applications/FilterBlock
  src/domain/thirdPartyApp/components/applications/MainTableThirdParty
  src/domain/thirdPartyApp/components/applications/TableView
  src/domain/thirdPartyApp/components/applications/TableView/SecureScore
  src/domain/thirdPartyApp/components/applications/index.tsx
  src/domain/thirdPartyApp/components/applications/AppCard/index.tsx
  src/domain/thirdPartyApp/components/applications/AppCard/Accounts/index.tsx
  src/domain/thirdPartyApp/components/applications/AppCard/Discovery/index.tsx
  src/domain/thirdPartyApp/components/applications/CardView/CardView.tsx
  src/domain/thirdPartyApp/components/applications/FilterBlock/index.tsx
  src/domain/thirdPartyApp/components/applications/MainTableThirdParty/index.tsx
  src/domain/thirdPartyApp/components/applications/TableView/index.tsx
  src/domain/thirdPartyApp/components/applications/TableView/SecureScore/index.tsx
```

### Component: `styled`
```
styled/

  src/domain/thirdPartyApp/components/styled/index.tsx
```
### Component: `AddCollaboratorDialog`
```
AddCollaboratorDialog/

  src/domain/Workspaces/components/AddCollaboratorDialog/index.tsx
```

### Component: `AddWorkspaceDialog`
```
AddWorkspaceDialog/

  src/domain/Workspaces/components/AddWorkspaceDialog/index.tsx
```

### Component: `ChangeCollaboratorRoleDialog`
```
ChangeCollaboratorRoleDialog/

  src/domain/Workspaces/components/ChangeCollaboratorRoleDialog/index.tsx
```

### Component: `DeleteWorkspaceDialog`
```
DeleteWorkspaceDialog/

  src/domain/Workspaces/components/DeleteWorkspaceDialog/index.tsx
```

### Component: `ExitWorkspaceDialog`
```
ExitWorkspaceDialog/

  src/domain/Workspaces/components/ExitWorkspaceDialog/index.tsx
```
### Component: `main`
```
main/

  src/domain/Workspaces/pages/main/index.tsx
```

### Component: `manageWorkspace`
```
manageWorkspace/

  src/domain/Workspaces/pages/manageWorkspace/index.tsx
```
### Component: `ActionsButton`
```
ActionsButton/

  src/shared/components/ActionsButton/index.tsx
```

### Component: `AppHeader`
```
AppHeader/

  src/shared/components/AppHeader/index.tsx
```

### Component: `Avatar`
```
Avatar/

  src/shared/components/Avatar/index.tsx
```

### Component: `BigButton`
```
BigButton/

  src/shared/components/BigButton/index.tsx
```

### Component: `ButtonWithBackground`
```
ButtonWithBackground/

  src/shared/components/ButtonWithBackground/index.tsx
```
### Component: `styles`
```
styles/
  the-golden-pyramid/src/components/styles/images
  the-golden-pyramid/src/components/styles/ABrick.styled.js
  the-golden-pyramid/src/components/styles/Acontainer.js
  the-golden-pyramid/src/components/styles/Body.js
  the-golden-pyramid/src/components/styles/Footer.styled.js
  the-golden-pyramid/src/components/styles/Header.styled.js
  the-golden-pyramid/src/components/styles/Row.styled.js
  the-golden-pyramid/src/components/styles/images/pyramids.jpg
```

---

## File Extensions in Use

```
 505 ts
 344 tsx
 161 svg
  65 graphql
  24 png
   6 js
   4 css
   1 txt
   1 pdf
   1 jsx
   1 jpg
   1 gql
```

---

## Config Files Present

- `package.json`

---

## Scaffolding Rules (enforce these every time)

1. **Never put styles directly in a component file** — all styling belongs in the `styled/` (or equivalent) sub-folder.
2. **Never put business logic directly in JSX** — extract to `hooks/` or `utils/`.
3. **One slice per component domain** — state lives in `redux/` (or `store/`) next to the component, not in a global monolith.
4. **Naming is PascalCase** — apply consistently to folders and files.
5. **Index barrel files** — each component folder exposes a clean public API via `index.ts` or `index.js`.
6. **New component = full sub-folder set** — even if a folder starts empty, create it. Consistency beats brevity.
7. **No cross-component imports from internals** — import from the index, never from deep paths like `ComponentA/redux/slice`.

---

## Anti-Patterns to Avoid

- Styles scattered in JSX `style={{}}` props
- Redux logic co-mingled with render logic
- Hooks that do styling; styled files that contain logic
- Flat component directories (no sub-folders)
- Deeply nested relative imports (`../../../`)
- Business logic in page/route components — delegate to feature components

---

## When Code Gets Deprecated

This skill is **structure-only**. When libraries or APIs change:
- Update the implementation files inside the sub-folders
- Keep the folder structure and naming conventions intact

---
