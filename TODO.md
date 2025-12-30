# TODO

## Deep Research & Approval Workflow

- [ ] Test Deep Research pipeline
  - [ ] Run with mock nodes: `python -m examples.07_deep_research.run "test topic" --mock --auto-approve`
  - [ ] Run with real Vel agents: `python -m examples.07_deep_research.run "AI safety trends" --auto-approve`
  - [ ] Verify all 3 research steps execute
  - [ ] Verify final report is generated

- [ ] Test human-in-the-loop via FE integration
  - [x] Create FastAPI endpoint for deep research (`/api/deep-research/*`)
  - [x] Implement session-based executor storage
  - [x] Register router in main.py
  - [x] Integrate `APPROVAL_PENDING` event handling in frontend
  - [x] Display approval data (plan title, steps) to user
  - [x] Implement approve/reject UI controls (ApprovalCard component)
  - [x] Connect approve/reject buttons to `POST /api/deep-research/approve`
  - [ ] Verify execution resumes after approval
  - [ ] Verify execution stops on rejection

## Backend Endpoints (mesh-app)

Endpoints created at `/Users/richard.s/mesh-app/backend/routers/deep_research.py`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/deep-research/execute` | POST | Start research, stream SSE until approval |
| `/api/deep-research/approve` | POST | Resume after user approves/rejects |
| `/api/deep-research/status/{session_id}` | GET | Check session status |
| `/api/deep-research/session/{session_id}` | DELETE | Cancel session |

## Frontend Integration Notes

When `data-approval-pending` event is received:
1. Parse `approval_data` from event (contains plan title, steps)
2. Show approval UI with plan details
3. User clicks approve/reject
4. POST to `/api/deep-research/approve` with `{ session_id, approved, rejection_reason? }`
5. Connect to new SSE stream from response
6. Continue rendering events until completion

## Frontend Files Modified

`/Users/richard.s/mesh-app/mesh-ui/src/components/ChatBubbleMeshChat.tsx`:
- Added `ApprovalState` and `ApprovalData` types
- Added `data-approval-pending` event handler
- Added `handleApprovalDecision`, `handleApprove`, `handleReject` functions
- Created `ApprovalCard` component with approve/reject UI
- Added approval card rendering when `approvalState.isPending` is true
- Disabled input during approval pending state
