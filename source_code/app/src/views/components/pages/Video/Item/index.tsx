import clsx from 'clsx';
import React, { Fragment, } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import commonCss from 'compDir/Theme/commonCss';
import { createStyles, Theme, makeStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiAlertOctagram, mdiCheck, mdiChevronDown, mdiChevronUp, mdiPencil, mdiRestore, mdiSend } from '@mdi/js';
import ColorButton from 'compDir/Button';
import UploadButton from 'compDir/Button/Upload';
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

export interface IStateToProps {
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any
}
interface IProps extends IDispatchToProps, IStateToProps {
	
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

// const VideoItem: FC<IProps> = (props: IProps) => {
const VideoItem: FC<IProps> = (props: IProps) => {
	const [open, setOpen] = React.useState(false);
	const classes = useStyles();

	return (
		<Fragment>
			<TableRow hover className={clsx(classes.root, classes.row, open ? 'active' : '')}>
				<TableCell>
					<Grid container spacing={1} direction='column' wrap='nowrap' alignItems='center'>{props.id}</Grid>
				</TableCell>
				<TableCell style={{ minWidth: 250 }}><Box>{props.title}</Box></TableCell>
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
								{
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
								}
								
								<Tooltip title="Stop">
									<ColorButton btnType="icon" btnColor="red"
										onClick={() => {  }}>
										<MDIcon size={'18px'} path={VideoStatus.STOPPED.iconName} />
									</ColorButton>
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
								
							</Grid>
						</Grid>
					</Collapse>
				</TableCell>
			</TableRow>
		</Fragment>
	);
};

export default VideoItem;