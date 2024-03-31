import React, { FC, Fragment, useRef, useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import ColorButton, { IProps as IColorBtnProps } from 'compDir/Button';
import { Grid, Chip, Tooltip, Box, FormHelperText } from '@material-ui/core';
import MDIcon from '@mdi/react';
import { mdiInformationOutline, mdiUpload } from '@mdi/js';
import { green, red } from '@material-ui/core/colors';
import { ApiRequest, Utils } from 'servDir';
import { Status } from 'constDir';
import { CircularProgress } from '@material-ui/core';
import { Movie, FiberNew } from '@material-ui/icons';
import Modal from 'compDir/Modal';
import { IFile } from 'interfaceDir';

const useStyles = makeStyles((theme) => ({
	input: {
		display: 'none',
	},
	chip: {
		maxWidth: '100%',
	},
	chipLabel: {
		paddingLeft: 0,
	},
	link: {
		border: 'none',
		background: 'none !important'
	},
}));

interface IProps extends IColorBtnProps {
	id?: any
	url?: string
	accept?: string
	multiple?: boolean
	displayOnly?: boolean
	fileInfo?: IFile
	dispType?: 'chip' | 'link'
	chipVariant?: 'default' | 'outlined'
	justify?: 'flex-start' | 'flex-end' | 'center'
	remoteDelete?: boolean
	extParams?: any
	maxWidth?: any
	onSuccess?(obj: any): any
	onDelete?(obj: any): any
	onError?(err: any, reason): any
	validateFile?(file: any): string
}

const UploadButton: FC<IProps> = (props: IProps) => {
	const classes = useStyles();
	const fileRef = useRef(null);
	const linkPrefix = `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/s3`;

	const { children,
		id = Math.random().toString(36).substr(2, 5),
		onSuccess = ((obj) => true),
		onError = ((err, reason) => true),
		onDelete = ((obj) => true),
		onClick = ((obj) => {
			if (fileUrl) {
				Utils.openNewTab(fileUrl);
			}
		}),
		validateFile = (file => null),
		extParams = {},
		uploadUrl = '/api/files/upload',
		accept = 'image/*,.zip',
		multiple = false,
		dispType = 'link',
		chipVariant = 'default',
		displayOnly = false,
		fileInfo,
		remoteDelete,
		maxWidth,
		...restProps
	} = props;

	const [isProcessing, setIsProcessing] = useState(false);
	const [confirmBox, setConfirmBox] = useState(null);
	// eg: my-file.csv
	const [fileName, setFileName] = useState();
	// eg: /task/21/process/5/2220220324_0912_mgpiyrebf.png
	const [filePath, setFilePath] = useState();
	// eg: https://stg-task.ca-adv.co.jp/s3/task/21/process/5/2220220324_0912_mgpiyrebf.png
	const [fileUrl, setUrl] = useState();
	const [fileStatus, setStatus] = useState(Status.SKIP);
	const [errMsg, setErrMsg] = useState<string | null>();

	// on state changed
	useEffect(() => {
		setFileName(fileInfo?.fileName);
		setFilePath(fileInfo?.filePath);
		// setStatus(Status.SKIP);
		setUrl(fileInfo?.filePath ? `${linkPrefix}/${fileInfo?.filePath}` : null);
		setErrMsg(null);
	}, [fileInfo]);

	const handleSelectFile = async (event) => {
		const file = event.target.files[0];
		if (file) {
			const errorMsg = validateFile(file);
			if (!errorMsg) {
				// upload file to S3
				await handleUploadFile(file);
			} else {
				setErrMsg(errorMsg);
				unSelectFile();
			}

		}
	};

	const handleUploadFile = (file) => {
		setProcessing(true);
		return ApiRequest.uploadFile(uploadUrl, file, extParams,
			(response) => {
				const respObj = response.data;
				setProcessing(false);

				if (!respObj.hasError) {
					const { fileName, filePath, url } = respObj.data;
					setFileName(fileName);
					setFilePath(filePath);
					setStatus(Status.NEW);
					setUrl(url);
					setErrMsg(null);


					return onSuccess({
						...respObj.data,
						status: Status.NEW
					});
				}
				return onError(respObj, 'upload:error');
			},
			(error) => {
				unSelectFile();
				console.error(error);
				setProcessing(false);
				setErrMsg(error.response.data.message);
				return onError(error, 'upload:exception');
			}
		);
	};

	const setProcessing = (processing) => {
		if (processing) {
			setErrMsg(null);
		}
		setIsProcessing(processing);
	};

	const unSelectFile = () => {
		fileRef.current.value = '';
		setFileName(null);
		setFilePath(null);
		setUrl(null);
		setStatus(null);
		setProcessing(false);
	};

	const removeFile = () => {
		unSelectFile();
		onDelete({ fileName, filePath, url: fileUrl, status: Status.DELETED, extParams });
	};

	const handleDeleteFile = () => {
		if (remoteDelete) {
			setProcessing(true);
			ApiRequest.sendPOST('/api/files/remove', extParams,
				(response) => {
					const respObj = response.data;

					if (!respObj.hasError) {
						removeFile();
						return;
					}
					setProcessing(false);
					return onError(respObj, 'delete:error');
				},
				(error) => {
					console.error(error);
					setProcessing(false);
					setErrMsg('Cannot delete file');
					return onError(error, 'delete:exception');
				}
			);
		} else {
			removeFile();
		}
	};

	const openConfirmBox = (callback: any) => {
		setConfirmBox({ callback });
	};

	const closeConfirmBox = (e, isConfirmed: boolean) => {
		const callback = confirmBox.callback;
		if (isConfirmed && typeof callback === 'function') {
			callback();
		}
		setConfirmBox(null);
	};

	const IconType = dispType == 'chip' ? (fileStatus == Status.NEW ? FiberNew : Movie) : Movie;
	const iconColor = dispType == 'chip' && fileStatus == Status.NEW ? { color: green[500] } : null;

	return (
		<Fragment>
			<Grid container>
				<input id={id} ref={fileRef} type="file" className={classes.input}
					accept={accept} multiple={multiple}
					onChange={handleSelectFile}
				/>
				{
					confirmBox && (
						<Modal fullWidth divider={false} maxWidth='xs'
							title='Delete confirmation'
							content={`Are you sure to delete this file「${fileName}」？`}
							handleClose={(e) => closeConfirmBox(e)}
							handleSubmit={(e) => closeConfirmBox(e, true)}
							submitLabel='Delete' closeLabel='Cancel'></Modal>
					)
				}
				{
					isProcessing ? (
						<Grid container alignContent='center' spacing={1}>
							<Grid item><CircularProgress size={18} /></Grid>
							<Grid item>
								<Box component='span' height={'100%'} fontWeight='fontWeightLight' fontSize={14}>対応中...</Box>
							</Grid>
						</Grid>
					) : (
						fileName ? (
							<Tooltip title={fileName}>
								<Chip size='small' variant={chipVariant}
									classes={{
										label: classes.chipLabel
									}}
									className={[classes.chip, dispType == 'chip' ? {} : classes.link]}
									style={maxWidth ? { maxWidth } : {}}
									icon={dispType == 'chip' ? (
										<IconType style={{ margin: 4, ...iconColor }} />
									) : (
										<Movie style={{ margin: 0 }} />
									)}
									label={<Box component='span'>{fileName}</Box>}
									onClick={onClick}
									onDelete={displayOnly ? undefined : () => remoteDelete ? openConfirmBox(handleDeleteFile) : handleDeleteFile()}
								/>
							</Tooltip>
						) : (
							<Fragment>
								{
									displayOnly ? (
										<Box component='span' height={'100%'} fontWeight='fontWeightLight' fontSize={13}>ファイルなし</Box>
									) : (
										<label htmlFor={id}>
											<ColorButton {...restProps} component="span" {...errMsg ? { btnColor: 'red' } : {}}
												startIcon={<MDIcon size={'16px'} path={mdiUpload} />}>
												{children}
											</ColorButton>
										</label>
									)
								}
							</Fragment>
						)
					)
				}
			</Grid>
			{
				errMsg && (
					<FormHelperText style={{ color: red[500], fontSize: '9pt', display: 'flex', alignItems: 'center', marginInline: 0 }}>
						<Box lineHeight="normal">
							<MDIcon style={{ verticalAlign: 'middle' }} size={'11pt'} path={mdiInformationOutline} />
							&nbsp;
							<Box component={'span'} style={{ verticalAlign: 'middle' }}>{errMsg}</Box>
						</Box>
					</FormHelperText>
				)
			}
		</Fragment>
	);
};

export default UploadButton;