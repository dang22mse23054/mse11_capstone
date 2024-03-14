import clsx from 'clsx';
import React, { Fragment, } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import commonCss from 'compDir/Theme/commonCss';
import { createStyles, Theme, makeStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiAlertOctagram, mdiCheck, mdiChevronDown, mdiChevronUp, mdiPencil, mdiRestore, mdiSend } from '@mdi/js';
import ColorButton from 'compDir/Button';
import UploadButton from 'compDir/Button/Upload';
import { IClient, ICAUser, IOrg, ITask, ITaskProcess } from 'interfaceDir';
import {
	Box, Divider, TableCell, TableRow, Link, Tooltip,
	Chip, Paper, Collapse, Grid, Typography, TextField
} from '@material-ui/core';
import {
	DestinationTypes, ScheduleTypes, TypeNames, TypeLabels, WeekLabels, DayOfWeekLabels,
	TaskProcStatus, FileTargets, TaskStatus, TIME_ZONE, TaskStatusMap, JobStatus, ProcessTypes
} from 'constDir';
import { toast } from 'material-react-toastify';
import { TaskService, Utils } from 'servDir';
import { TaskProcessInput } from 'inputModelDir';

import moment from 'moment-timezone';
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup';
import ToggleButton from '@material-ui/lab/ToggleButton';
moment.tz.setDefault(TIME_ZONE);

export interface IStateToProps {
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any
}
const unknow = '不明';
interface IProps extends ITask, IDispatchToProps, IStateToProps {
	onChange?(newProps: any): void
	onClickEdit(): any
	reloadPage?(): any
	beforeSubmit?(type: '追加' | '編集' | '対応なし' | '完了' | '戻る', callback: any): any
}

const useStyles = makeStyles((theme: Theme) => createStyles({
	...commonCss,
	...tableCss,
	root: {
		'& > *': {
			borderBottom: 'unset',
		},
	},
	subInfo: {
		background: '#FFFFFF',
		padding: theme.spacing(2)
	},
	
}));

// const TaskItem: FC<IProps> = (props: IProps) => {
const TaskItem: FC<IProps> = (props: IProps) => {
	const [open, setOpen] = React.useState(false);
	const [processType, setProcessType] = React.useState(0);
	const classes = useStyles();
	const [sendingProcs, setSendingProcs] = React.useState(new Map<number, boolean>());
	const [sendingTask, setSendingTask] = React.useState(false);

	const handleChangeProcTime = (event: React.ChangeEvent<HTMLInputElement>, processId: string) => {
		const value = Number(event.target.value);
		if (value >= 0) {
			const newProps = { ...props };
			const target = newProps.processes.filter(process => process.id == processId)[0];
			target.processingTime = value;
			validateProcess(target, ['processingTime']);
			if (props.onChange) {
				props.onChange(newProps);
			}
		} else {
			event.target.value = 0;
		}
	};

	// Container of process filtered by procType
	const processContainer = props.processes.filter((process) => process.procType == processType);
	const onlyProcessContainer = props.processes.filter((process) => process.procType == ProcessTypes.PROCESS.value);
	const mustDoneBeforeSubmit = props.processes;

	let countSubmitProc = 0;
	// if have any 完了 proc and no have proc 未完了 then disable 送付 button
	let disabledSendTask = false;
	// if have any 完了 proc then disable task's 対応なし button
	let disabledNoCorres = false;
	for (let i = 0; i < mustDoneBeforeSubmit?.length; i++) {
		// disable 対応なし
		if (mustDoneBeforeSubmit[i].status == TaskProcStatus.CLOSED.value) {
			disabledNoCorres = true;
		}

		// check have  proc 未完了
		if (mustDoneBeforeSubmit[i].status == TaskProcStatus.OPEN.value) {
			countSubmitProc++;
		}
	}
	disabledSendTask = !disabledNoCorres || (countSubmitProc > 0);

	const reloadTask = async () => {
		if (props.onChange) {
			const taskId = Number(props.id);
			props.onChange(await TaskService.getTask(taskId));
		}
	};

	return (
		<Fragment>
			<TableRow hover className={clsx(classes.root, classes.row, open ? 'active' : '')}>
				<TableCell>
					<Grid container spacing={1} direction='column' wrap='nowrap' alignItems='center'>
						<Chip size="small" className={props.type == ScheduleTypes.SPOT.value ? classes.spotChip : classes.regularChip}
							label={(
								<Box>
									{props.scheduleId}
									{props.oldScheduleId && (
										<Tooltip title='旧ID'>
											<Box component='span' textAlign="center" paddingLeft={1} fontSize={9} fontWeight="fontWeightLight" fontStyle="italic">({props.oldScheduleId})</Box>
										</Tooltip>
									)}
									<Box component='span' textAlign="center" paddingLeft={1} fontSize={9} fontWeight="fontWeightLight" fontStyle="italic">v.{props.destVersion}</Box>
								</Box>
							)} />
						<Grid item>
							<Box component='div' textAlign="center" >
								<Tooltip title='タスクID'>
									<Box component='span' fontSize={9} fontWeight="fontWeightLight">#{props.id}</Box>
								</Tooltip>
								{props.oldId && (
									<Tooltip title='旧タスクID'>
										<Box component='span' fontSize={9} fontWeight="fontWeightLight" fontStyle="italic"> ({props.oldId})</Box>
									</Tooltip>
								)}
							</Box>
						</Grid>
					</Grid>
				</TableCell>
				<TableCell><Box>{props.name}</Box></TableCell>
				<TableCell style={{ minWidth: 180 }}>
					<Box whiteSpace='pre-wrap'>
						{
							props.destinations.map((dest, idx) => {

								switch (props.destType) {
									case DestinationTypes.CLIENT.value:
										const client = (dest as IClient);
										if (!client) {
											return `■ ${unknow}\n`;
										}
										return `■ ${client.clientCompanyName}\n ⎩ ${client.name}\n`;

									case DestinationTypes.DIVISION.value:
										const org = (dest as IOrg);
										if (!org) {
											return `◉ ${unknow}\n`;
										}
										const parts = org.mapText.split('>');
										return parts.map((item, idx) => `${idx > 0 ? ' ⎩ ' : '◉ '}${parts[idx]}`).join('\n') + '\n';

									case DestinationTypes.HANDLE_USER.value:
										const user = (dest as ICAUser);
										if (!user) {
											return `• ${unknow}\n`;
										}
										return (
											<Grid container spacing={1} key={idx}>
												<Grid item xs={12} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
													• <Tooltip title={`${user.fullname} (${user.uid})`}>
														<Link target="_blank" rel="noopener" href={`https://telma.caad.isca.jp/employee/${user.uid}`}>{user.fullname}</Link>
													</Tooltip>
												</Grid>
											</Grid>
										);

									default:
										return '-';
								}
							})
						}
					</Box>
				</TableCell>
				<TableCell align='center'>
					<Box>
						{
							props.caDashboards ?
								props.caDashboards.split(',').map((val, index) => `${index == 0 ? '' : ', '}${val}`)
								: '-'
						}
					</Box>
				</TableCell>
				<TableCell>
					<Box whiteSpace='pre-wrap'>{`${props.category?.parent?.name || '不明な大カテゴリ'}\n(${props.category?.name || '不明な中カテゴリ'})`}</Box>
				</TableCell>
				<TableCell>
					<Box whiteSpace='pre-wrap'>
						{
							props.media && props.media.length > 0 ?
								props.media.map((media, index) => `${index == 0 ? '' : ', '}${media?.name || '不明'}`)
								: '-'
						}
					</Box>
				</TableCell>

				<TableCell style={{ minWidth: 90 }}>
					<Box whiteSpace='pre-wrap'>
						{
							moment(new Date(parseInt(props.deadline))).format('YYYY/MM/DD[\n]HH:mm')
						}
					</Box>
				</TableCell>
				{/* Show process number */}
				<TableCell style={{ minWidth: 70 }} align='center'>
					<Tooltip title={TaskStatusMap[props.status].label}>
						<Grid container alignContent='center' justify='space-around'>
							<MDIcon style={{ verticalAlign: 'middle', color: TaskStatusMap[props.status].iconFixedColor }}
								size={'13pt'} path={TaskStatusMap[props.status].iconName} />
							<Box>
								{
									onlyProcessContainer ? (
										`${onlyProcessContainer.filter((process) => process.status == TaskProcStatus.CLOSED.value).length}/${onlyProcessContainer.length}`
									) : '-'
								}
							</Box>
						</Grid>
					</Tooltip>
				</TableCell>
				{/* Task Owner */}
				<TableCell>
					{props.owner ? (
						<Grid container spacing={1}>
							<Grid item xs={12} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
								<Tooltip title={`${props.owner.fullname} (${props.owner.uid})`}>
									<Link target="_blank" rel="noopener" href={`https://telma.caad.isca.jp/employee/${props.owner.uid}`}>{props.owner.fullname}</Link>
								</Tooltip>
							</Grid>
						</Grid>
					) : '不明'}
					{
						props.executor && props.executor.id != props.owner?.id && (
							<Grid container spacing={1} direction='column' wrap='nowrap'>
								<Grid item style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
									<Box fontSize={11} fontWeight="fontWeightLight">
										(実行者：<Tooltip title={`${props.executor.fullname} (${props.executor.uid})`}>
											<Link color='secondary' target="_blank" rel="noopener" href={`https://telma.caad.isca.jp/employee/${props.executor.uid}`}>
												{props.executor.fullname}
											</Link>
										</Tooltip>)
									</Box>
								</Grid>
							</Grid>
						)
					}
				</TableCell>
				{/* Upload button */}
				<TableCell style={{ minWidth: 140 }}>
					{
						props.status != TaskStatus.SKIPED.value && (
							<UploadButton size='small' variant='outlined' btnColor='secondary'
								displayOnly={props.status != TaskStatus.OPEN.value}
								dispType='link'
								accept='*.*'
								remoteDelete
								fileInfo={{
									fileName: props.fileName,
									filePath: props.filePath
								}}
								extParams={{
									target: FileTargets.Task,
									taskId: props.id,
									updatedAt: props.updatedAt,
								}}
								onSuccess={reloadTask}
								onDelete={reloadTask}
							>ファイル追加</UploadButton>
						)
					}
				</TableCell>
				<TableCell align='center' style={{ minWidth: 90 }}>
					{
						props.status == TaskStatus.OPEN.value ? (
							// SUBMIT BUTTON
							<ColorButton variant="contained" size="small"
								disabled={disabledSendTask || sendingTask}
								endIcon={<MDIcon size={'16px'} path={mdiSend} />}
								onClick={() => {
									props.beforeSubmit('完了', () => {
										changeTaskStatus(TaskStatus.CLOSED.value);
									});
								}}>
								送付
							</ColorButton>
						) : (
							<Grid container spacing={1} direction='column'>
								<Grid item>
									<ColorButton variant="outlined" size="small" btnColor='secondary'
										disabled={sendingTask}
										startIcon={<MDIcon size={'13px'} path={mdiRestore} />}
										onClick={() => {
											props.beforeSubmit('再開', () => {
												changeTaskStatus(TaskStatus.OPEN.value);
											});
										}}>
										戻る
									</ColorButton>
								</Grid>
								{
									props.snsStatus != null && props.snsStatus < JobStatus.DONE && (
										<Grid item>
											{
												props.snsStatus == JobStatus.ERROR ? (
													<Tooltip title={(
														<Fragment>
															{props.destinations?.length > 1 ? '一部の' : ''}CWの通知が失敗しました<br />
															再通知のためにこちらをクリックしてください
														</Fragment>
													)}>
														<Link color='error' underline={'hover'} style={{ cursor: 'pointer' }}
															onClick={renotify}>
															<Grid container justify='center' alignItems='center' spacing={1}>
																<MDIcon size={'13px'} path={mdiAlertOctagram} />
																<Grid item>再通知</Grid>
															</Grid>
														</Link>
													</Tooltip>
												) : JobStatus.SKIPED == props.snsStatus ? (
													<Fragment></Fragment>
												) : (
													<Box color='primary.main' >{'通知中...'}</Box>
												)
											}
										</Grid>
									)
								}
							</Grid>
						)
					}
				</TableCell>
				<TableCell>
					<ColorButton size='small' btnType='icon' onClick={() => setOpen(!open)}>
						<MDIcon size={'24px'} path={open ? mdiChevronUp : mdiChevronDown} />
					</ColorButton>
				</TableCell>
			</TableRow>

			{/* Task Detail */}
			<TableRow className={clsx(classes.row, 'active')}>
				<TableCell colSpan={12} className={open ? '' : 'close'}>
					<Collapse in={open} timeout="auto" unmountOnExit>
						<Grid className={clsx(classes.subInfo)}>
							<Grid container spacing={2}>
								{/* General Info */}
								<Grid item xs={4}>
									<Grid container spacing={1} direction='column'>
										<Grid item><Typography variant="subtitle2">スケジューラ</Typography></Grid>
										<Grid item>
											<Paper className={clsx(classes.taskDetail)} elevation={0}>
												<Grid container spacing={1}>
													<Grid item container>
														<Grid item xs={3}><Box color='text.secondary' fontWeight='fontWeightBold'>頻度</Box></Grid>
														<Grid item xs={9}>
															<Grid container spacing={1} alignItems='baseline'>
																<Grid item>
																	<Chip label={props.type == ScheduleTypes.SPOT.value ? ScheduleTypes.SPOT.label : ScheduleTypes.REGULAR.label} size="small"
																		className={props.type == ScheduleTypes.SPOT.value ? classes.spotChip : classes.regularChip} />
																</Grid>
																{
																	(props.type == 2) && (
																		<Grid item style={{ flex: 1 }}>
																			<Grid container>
																				<Grid item>{`${TypeLabels[props.frequency.type]}：`}</Grid>
																				<Grid item style={{ flex: 1 }}>
																					{(() => {
																						switch (props.frequency.type) {
																							case TypeNames.WEEKLY:
																								return (
																									props.frequency.dayOfWeek?.split(',').map(data => DayOfWeekLabels[data]).join(', ')
																								);
																							case TypeNames.MONTHLY_BY_WEEK:
																								return (
																									<Fragment>
																										{props.frequency.weekOfMonth?.split(',').map(data => WeekLabels[data]).join(', ')}
																										<Box>{props.frequency.dayOfWeek?.split(',').map(data => DayOfWeekLabels[data]).join(', ')}</Box>
																									</Fragment>
																								);
																							case TypeNames.MONTHLY_BY_DATE:
																								return (
																									`${props.frequency.dates}${props.frequency.isOnlyBusinessDays ? '営業日' : '日'}`
																								);
																						}
																					})()}
																					<Box component='span'>{props.frequency?.isSkipHoliday && '（祝日を除く）'}</Box>
																				</Grid>
																			</Grid>
																		</Grid>
																	)
																}
															</Grid>
														</Grid>
													</Grid>
													<Grid item container>
														<Grid item xs={3}><Box color='text.secondary' fontWeight='fontWeightBold'>参考ファイル</Box></Grid>
														<Grid item xs={9}>
															{
																props.refFileName ? (
																	<Link underline={'hover'} style={{ cursor: 'pointer' }}
																		onClick={() => {
																			if (props.refFilePath) {
																				Utils.openNewTab(`https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/s3/${props.refFilePath}`);
																			}
																		}}>
																		{props.refFileName}
																	</Link>
																) : (
																	<Box color='primary.main' >なし</Box>
																)
															}
														</Grid>
													</Grid>
													<Grid item container>
														<Grid item xs={3}><Box color='text.secondary' fontWeight='fontWeightBold'>メモ</Box></Grid>
														<Grid item xs={9}><Box component='pre' margin={0} whiteSpace='pre-wrap'>{props.memo || '-'}</Box></Grid>
													</Grid>
												</Grid>
											</Paper>
										</Grid>
									</Grid>
								</Grid>
								{/* Process List */}
								<Grid item xs={8}>									
									<Grid container spacing={2} direction='column'>
										{/* Table Header */}
										<Grid item container>
											<Grid item lg={6} sm={4}>
												{/*<Typography variant="subtitle2">プロセス</Typography>*/}
												<ToggleButtonGroup 
													color="primary"
													size="small" 
													exclusive 
													onChange={(event, value)=> setProcessType(value)} 
													value={processType}
												>
													{
														Object.keys(ProcessTypes).map((key, index) => (
															<ToggleButton
																style={{ textTransform: 'none' }}
																{...processType == ProcessTypes[key].value && {
																	disabled: true,
																	style: { color: 'inherit', textTransform: 'none' }
																}}
																key={index} value={ProcessTypes[key].value} >
																{ProcessTypes[key].label}
															</ToggleButton>
														))
													}
												</ToggleButtonGroup>
											</Grid>
											<Grid item lg={6} sm={8}>
												<Grid container justify='flex-end' spacing={2}>
													<Grid item style={{ alignSelf: 'center' }}>
														<Grid container justify='flex-end' spacing={2} alignItems='baseline'>
															<Grid item>
																<Typography component='div' variant="subtitle2">
																	<Box fontSize={12} fontWeight="fontWeightBold" color="text.secondary">実行時間TOTAL</Box>
																</Typography>
															</Grid>
															<Grid item>
																<Typography component='div' variant="body2">
																	<Box width={70} textAlign='center'>{props.processingTime || props.esimatedTime || '- '}分</Box>
																</Typography>
															</Grid>
														</Grid>
													</Grid>
													{
														props.status == TaskStatus.OPEN.value ? (
															<Fragment>
																<Grid item>
																	<ColorButton btnColor='secondary' size="small" variant="outlined"
																		disabled={sendingTask}
																		startIcon={<MDIcon size={'13px'} path={mdiPencil} />}
																		onClick={props.onClickEdit}>
																		編集
																	</ColorButton>
																</Grid>
																<Grid item style={{ minWidth: 90 }}>
																	<ColorButton btnColor='red' size="small" variant="outlined"
																		disabled={disabledNoCorres || sendingTask}
																		onClick={() => {
																			props.beforeSubmit('対応なし', () => {
																				changeTaskStatus(TaskStatus.SKIPED.value);
																			});
																		}}>
																		対応なし
																	</ColorButton>
																</Grid>
															</Fragment>
														) : (
															<Fragment></Fragment>
														)
													}
												</Grid>
											</Grid>
										</Grid>
										{/* BEGIN -- Process Item */}
										{
											// DEBUG: Loading processes of task
											// Implement new code for check list
											processContainer?.length ? processContainer.map((process, id) => {
												// prevent edit processing time if Task is SKIPED or TaskProc is Not OPEN
												const disabledProcessingTime = (props.status == TaskStatus.SKIPED.value || process.status != TaskProcStatus.OPEN.value);
												// set default processingTime (to only click 完了 button without edit processingTime)
												process.processingTime = process.processingTime || process.estimatedTime;
												// If checklist (procType > 0) then no counting processing time. Overwrite data.
												if (process.procType > 0) {
													process.processingTime = 0;
												}
												const processingTime = process.processingTime || '';
												const isCheckListProcess = process.procType !== ProcessTypes.PROCESS.value;

												return (
													<Grid key={id} item container spacing={2} direction='column'>
														<Grid item><Divider /></Grid>
														<Grid item container alignItems='center' spacing={1}>
															{/* Process Name */}
															<Grid item lg={3} sm={2}>
																<Tooltip title='主プロセス' disableHoverListener={!process.isMain} placement="bottom-start">
																	<Grid container alignItems='center'>
																		{/* {process.isMain && <StarRate color='primary' />} */}
																		<Box fontWeight={(process.isMain || process.isRequired) ? 'fontWeightBold' : 'fontWeightRegular'}
																			{...process.isMain && { color: 'primary.main' }}>{process.name}</Box>
																	</Grid>
																</Tooltip>

															</Grid>
															{/* <Grid item lg={3} sm={2}><Box>{process.owner.fullname} ({process.owner.uid})</Box></Grid> */}
															{/* Process Owner */}
															<Grid item lg={3} sm={2}>
																{process.owner ? (
																	<Grid container spacing={1} direction='column' wrap='nowrap'>
																		<Grid item style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
																			<Tooltip title={`${process.executor ? '担当者：' : ''}${process.owner.fullname} (${process.owner.uid})`}>
																				<Link color='secondary' target="_blank" rel="noopener" href={`https://telma.caad.isca.jp/employee/${process.owner.uid}`}>
																					{process.owner.fullname}
																				</Link>
																			</Tooltip>
																		</Grid>
																	</Grid>
																) : '不明な担当者'}
																{
																	process.executor && process.executor.id != process.owner?.id && (
																		<Grid container spacing={1} direction='column' wrap='nowrap'>
																			<Grid item style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
																				<Box fontSize={11} fontWeight="fontWeightLight">
																					(実行者：<Tooltip title={`${process.executor.fullname} (${process.executor.uid})`}>
																						<Link color='secondary' target="_blank" rel="noopener" href={`https://telma.caad.isca.jp/employee/${process.executor.uid}`}>
																							{process.executor.fullname}
																						</Link>
																					</Tooltip>)
																				</Box>
																			</Grid>
																		</Grid>
																	)
																}
															</Grid>
															<Grid item lg={6} sm={8}>
																<Grid container justify='flex-end' alignItems='center' spacing={2} wrap='nowrap'>
																	<Grid item>
																		<Grid container alignItems='center' spacing={1} wrap='nowrap'>
																			<Grid item style={{ maxWidth: 140 }}>
																				{/* Upload button */}
																				{
																					process.status != TaskProcStatus.SKIPED.value && props.status != TaskStatus.SKIPED.value && (
																						<UploadButton size='small' variant='outlined' btnColor='secondary'
																							displayOnly={process.status != TaskProcStatus.OPEN.value}
																							dispType='link'
																							accept='*.*'
																							remoteDelete
																							extParams={{
																								target: FileTargets.TaskProcess,
																								taskId: process.taskId,
																								taskProcId: process.id,
																								updatedAt: process.updatedAt,
																							}}
																							fileInfo={{
																								fileName: process.fileName,
																								filePath: process.filePath
																							}}
																							onSuccess={reloadTask}
																							onDelete={reloadTask}
																							onError={(err) => {
																								if (err.response?.data?.code == 10005) {
																									reloadTask();
																								}
																							}}
																						>ファイル追加</UploadButton>
																					)
																				}
																			</Grid>
																			<Grid item>
																				{/* Process Estimated Time */}
																				<Box width={70} textAlign='center'>
																					{
																						process.status == TaskProcStatus.SKIPED.value ? (
																							<Box fontWeight='fontWeightRegular'>対応なし</Box>
																						) : (
																							<TextField
																								disabled={disabledProcessingTime || (process.procType > 0)}
																								type='number'
																								size='small'
																								placeholder="-"
																								inputProps={{ style: { textAlign: 'center' }, min: 0, max: 1000 }}
																								InputProps={{
																									endAdornment: (
																										<Box paddingBottom={'5px'} color='text.secondary' fontWeight="fontWeightLight" >分</Box>
																									),
																								}}
																								value={processingTime}
																								onFocus={(event) => event.target.select()}
																								onChange={(event) => { handleChangeProcTime(event, process.id); }}
																								error={Boolean(process.processingTimeError)}
																								// helperText={process.processingTimeError}
																							/>
																						)
																					}

																				</Box>
																			</Grid>
																		</Grid>
																	</Grid>

																	{
																		// If task is closed or skipped then hide these buttons: closed and skipped
																		[TaskStatus.CLOSED.value, TaskStatus.SKIPED.value].includes(props.status) ? (
																			<Fragment></Fragment>
																		) : (
																			<Fragment>
																				{
																					// Case: task status and process status are opened.
																					process.status == TaskProcStatus.OPEN.value ? (
																						<Fragment>
																							{/* Process Button */}
																							<Grid item /* style={{ width: 90 }} */>
																								<ColorButton size="small" variant='contained' btnColor="secondary"
																									startIcon={<MDIcon size={'13px'} path={mdiCheck} />}
																									{...(sendingProcs.get(process.id)) ? {
																										disabled: true, onClick: undefined
																									} : {
																										onClick: () => {
																											const submitProc = {
																												...process,
																												status: TaskProcStatus.CLOSED.value,
																											};
																											if (validateProcess(submitProc, ['processingTime', 'id'])) {
																												changeProcStatus(submitProc);

																											} else {
																												const newProps = { ...props };
																												newProps.processes[id] = {
																													...process,
																													processingTimeError: submitProc.processingTimeError,
																												};
																												if (props.onChange) {
																													props.onChange(newProps);
																												}
																											}
																										}
																									}}>
																									完了
																								</ColorButton>
																							</Grid>
																							<Grid item style={{ minWidth: 90 }}>
																								<ColorButton size="small" variant="outlined" btnColor="secondary"
																									{...(sendingProcs.get(process.id)) ? {
																										disabled: true, onClick: undefined
																									} : {
																										onClick: () => {
																											const submitProc = {
																												...process,
																												processingTime: null,
																												status: TaskProcStatus.SKIPED.value,
																											};
																											if (validateProcess(submitProc, ['id'])) {
																												changeProcStatus(submitProc);

																											}
																										},
																										disabled: isCheckListProcess
																									}}>
																									対応なし
																								</ColorButton>
																							</Grid>
																						</Fragment>
																					) : (
																						// Case: task status opened and process status closed.
																						<Fragment>
																							<Grid item /* style={{ width: 90 }} */>
																								<ColorButton size="small" variant='outlined' btnColor='secondary'
																									startIcon={<MDIcon size={'13px'} path={mdiRestore} />}
																									{...(sendingProcs.get(process.id)) ? {
																										disabled: true, onClick: undefined
																									} : {
																										onClick: () => {
																											const submitProc = {
																												...process,
																												status: TaskProcStatus.OPEN.value,
																											};
																											if (validateProcess(submitProc, ['id'])) {
																												changeProcStatus(submitProc);

																											}
																										}
																									}}>
																									戻る
																								</ColorButton>
																							</Grid>
																							<Grid item style={{ minWidth: 90 }}>
																							</Grid>
																						</Fragment>
																					)
																				}
																			</Fragment>
																		)
																	}
																</Grid>
															</Grid>
														</Grid>
													</Grid>
												);
											}) : (
												// No checklist
												<Grid item container spacing={2} direction='column'>
													<Grid item><Divider /></Grid>
													<Grid item><Box textAlign='center'>アイテムはありません</Box></Grid>
												</Grid>
											)
										}
										{/* END -- Process Item */}
									</Grid>
								</Grid>
							</Grid>

						</Grid>
					</Collapse>
				</TableCell>
			</TableRow>
		</Fragment>
	);
};

export default TaskItem;