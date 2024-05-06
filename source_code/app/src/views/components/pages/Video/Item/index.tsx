import clsx from 'clsx';
import React, { Fragment, } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import commonCss from 'compDir/Theme/commonCss';
import { createStyles, Theme, makeStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiAlertOctagram, mdiCheck, mdiChevronDown, mdiChevronUp, mdiCogOutline, mdiPencil, mdiRestore, mdiSend } from '@mdi/js';
import ColorButton from 'compDir/Button';
import UploadButton from 'compDir/Button/Upload';
import Chart from 'compDir/Chart';
import {
	Box, Divider, TableCell, TableRow, Link, Tooltip,
	Chip, Paper, Collapse, Grid, Typography, TextField
} from '@material-ui/core';
import {
	TIME_ZONE,
	VideoStatus,
	VideoStatusMap,

} from 'constDir';

import moment from 'moment-timezone';
moment.tz.setDefault(TIME_ZONE);

import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import video from 'rootDir/services/validation/video';
import { blue } from '@material-ui/core/colors';



export interface IStateToProps {
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any
}
interface IProps extends IDispatchToProps, IStateToProps {
	onChangeStatus: (video: any) => void;
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

const createData = (data) => {
	return {
		group: data.group,
		videoId: data.videoId,
		date: new Date(data.createdAt),

		total: data.genderDetail.male + data.genderDetail.female,
		total_happy: data.happyDetail.gender.male + data.happyDetail.gender.female,

		children: data.ageDetail.children,
		happy_children: data.happyDetail.age.children,
		
		teenager: data.ageDetail.teenager,
		happy_teenager: data.happyDetail.age.teenager,
		
		adult: data.ageDetail.adult,
		happy_adult: data.happyDetail.age.adult,
		
		middleAge: data.ageDetail.middleAge,
		happy_middleAge: data.happyDetail.age.middleAge,
		
		elderly: data.ageDetail.elderly,
		happy_elderly: data.happyDetail.age.elderly,
		
		male: data.genderDetail.male,
		happy_male: data.happyDetail.gender.male,
		
		female: data.genderDetail.female,
		happy_female: data.happyDetail.gender.female,
	}
}

// const VideoItem: FC<IProps> = (props: IProps) => {
const VideoItem: FC<IProps> = (props: IProps) => {
	const [open, setOpen] = React.useState(false);
	const classes = useStyles();

	const targetStatusBtn = props.status == VideoStatus.PLAYING.value ? VideoStatus.PAUSED : VideoStatus.PLAYING;
	let statistic = (props.statistic || []).map((data) => createData(data));

	// statistic = [
	// 	{ name: '202020204', total: 25, total_happy: 20, children: 0, happy_children: 20, teenager:30, happy_teenager: 40, adult: 18, happy_adult: 16, middleAge: 15, happy_middleAge: 12, elderly: 12, happy_elderly: 12, male: 11, happy_male: 9, female: 14, happy_female: 11, },
	// ];
	return (
		<Fragment>
			<TableRow hover className={clsx(classes.root, classes.row, open ? 'active' : '')}>
				<TableCell>
					<Grid container spacing={1} direction='column' wrap='nowrap' alignItems='center'>{props.id}</Grid>
				</TableCell>
				<TableCell style={{ minWidth: 250 }}><Box>{props.title}</Box></TableCell>
				<TableCell style={{ minWidth: 250 }}>
					<UploadButton size='small' variant='outlined' btnColor='secondary'
						displayOnly={true}
						dispType='link'
						accept='*.*'
						onClick={props.onReviewVideo}
						fileInfo={{
							fileName: props.refFileName,
							filePath: props.refFilePath
						}}
					>ファイル追加</UploadButton>

				</TableCell>
				<TableCell style={{ minWidth: 200 }}>
					{
						props.categories && props.categories.length > 0 ?
							props.categories.map((categories, index) => `${index == 0 ? '' : ', '}${categories?.name || 'Unknown'}`)
							: '-'
					}
				</TableCell>
				<TableCell>
					<Box whiteSpace='pre-wrap'>
						{
							moment(new Date(parseInt(props.createdAt))).format('YYYY/MM/DD[\n]HH:mm')
						}
					</Box>
				</TableCell>
				<TableCell>
					<Grid container alignContent='center'>
						<MDIcon style={{ verticalAlign: 'middle', color: VideoStatusMap[props.status].iconFixedColor }}
							size={'13pt'} path={VideoStatusMap[props.status].iconName} />
						<Box paddingLeft={0.3}>{VideoStatusMap[props.status].label}</Box>
					</Grid>
				</TableCell>
				<TableCell align='center'>
					{
						props.status != VideoStatus.STOPPED.value && (
							<Fragment>
								{/* {
									props.status == VideoStatus.PLAYING.value ? (
										<Fragment>
											<Tooltip title="Pause">
												<ColorButton btnType="icon" btnColor="grey"
													onClick={() => { }}>
													<MDIcon size={'18px'} path={VideoStatus.PAUSED.iconName} />
												</ColorButton>
											</Tooltip>
										</Fragment>
									) : (
										<Tooltip title="Play">
											<ColorButton btnType="icon" btnColor="blue"
												onClick={() => { }}>
												<MDIcon size={'18px'} path={VideoStatus.PLAYING.iconName} />
											</ColorButton>
										</Tooltip>
									)
								} */}
								<Tooltip title={targetStatusBtn.btnLabel}>
									<ColorButton btnType="icon" 
										btnColor={targetStatusBtn == VideoStatus.PLAYING ? 'blue' : 'grey'}
										onClick={() => {
											props.onChangeStatus({
												...props,
												isEnabled: targetStatusBtn == VideoStatus.PLAYING,
											});
										}}>
										<MDIcon size={'18px'} path={targetStatusBtn.iconName} />
									</ColorButton>
								</Tooltip>
								
								<Tooltip title="Stop">
									<ColorButton btnType="icon" btnColor="red"
										onClick={() => { 
											props.onStop({
												...props,
												isEnabled: targetStatusBtn == VideoStatus.PLAYING,
											});
										}}>
										<MDIcon size={'18px'} path={VideoStatus.STOPPED.iconName} />
									</ColorButton>
								</Tooltip>

								<Tooltip title="Edit">
									<Box component='span'>
										<ColorButton btnType="icon" btnColor="secondary"
											disabled={props.status == VideoStatus.PLAYING.value}
											onClick={props.onClickEdit}>
											<MDIcon size={'18px'} path={mdiCogOutline} />
										</ColorButton>
									</Box>
								</Tooltip>
							</Fragment>
						)
					}
				</TableCell>
				<TableCell>
					<Grid container justify='flex-end'>
						<ColorButton size='small' btnType='icon' onClick={() => setOpen(!open)}>
							<MDIcon size={'24px'} path={open ? mdiChevronUp : mdiChevronDown} />
						</ColorButton>
					</Grid>
				</TableCell>
			</TableRow>

			{/* Task Detail */}
			<TableRow className={clsx(classes.row, 'active')}>
				<TableCell colSpan={12} className={open ? '' : 'close'}>
					<Collapse in={open} timeout="auto" unmountOnExit>
						<Grid className={clsx(classes.subInfo)}>
							<Grid container spacing={2}>
								<Grid item xs={4}>
									<Chart data={statistic}/>
								</Grid>
								<Grid item xs={8}>	
									<TableContainer component={Box} style={{minHeight: 250, maxHeight: 400}}>
										<Table size="small" stickyHeader aria-label="sticky table">
											<caption style={{textAlign:'right'}}>
												Unit: (<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>happy</Box>/total)
											</caption>
											<TableHead>
												<TableRow>
													<TableCell>Group</TableCell>
													<TableCell align="right">View</TableCell>
													<TableCell align="right">male</TableCell>
													<TableCell align="right">female</TableCell>
													<TableCell align="right">children</TableCell>
													<TableCell align="right">teenager</TableCell>
													<TableCell align="right">adult</TableCell>
													<TableCell align="right">middleAge</TableCell>
													<TableCell align="right">elderly</TableCell>
												</TableRow>
											</TableHead>
											<TableBody>
											{statistic.map((row, index) => (
												<TableRow key={index}>
													<TableCell component="th" scope="row">
														{row.group}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.total_happy}</Box>/{row.total}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_male}</Box>/{row.male}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_female}</Box>/{row.female}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_children}</Box>/{row.children}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_teenager}</Box>/{row.teenager}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_adult}</Box>/{row.adult}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_middleAge}</Box>/{row.middleAge}
													</TableCell>
													<TableCell align="right">
														<Box component='span' fontWeight="fontWeightBold" color={blue[600]}>{row.happy_elderly}</Box>/{row.elderly}
													</TableCell>
												</TableRow>
											))}
											</TableBody>
										</Table>
									</TableContainer>
								</Grid>								
							</Grid>


							
						</Grid>
					</Collapse>
				</TableCell>
			</TableRow>
		</Fragment>
	);
};

export default VideoItem;